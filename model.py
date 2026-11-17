"""
SDIQA Model
============

Official implementation of SDIQA for quality assessment of AI-generated
images in industrial design.

Architecture
------------
    - Visual Backbone   : Swin Transformer V2-B (ImageNet-1K pretrained, trainable)
    - Text Encoder      : OpenCLIP ViT-B-32-quickgelu (LAION-400M pretrained, frozen)
    - Physical Stream   : GPU-based illumination/material/aesthetic feature extraction
                          + Physics-Gated Fusion (PGF) module, consisting of
                            PV-OPF, Visual Projection and Gated Residual Blending
    - Semantic  Stream  : Cross-Modal Attention
                          + residual bottleneck projections
    - Quality Regression: 5 parallel QR heads, one per evaluation dimension

The five output dimensions follow the order:
    [text_image_consistency, material, light, creativity, aesthetics]
"""

import math
import logging
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

logger = logging.getLogger(__name__)


# ============================================================
# Physical Feature Extractors (GPU-only, no trainable params)
# ============================================================

class LightFeatureExtractor(nn.Module):
    """Extract 4-D light statistics on GPU.

    Output (B, 4):
        [mean_brightness, std_brightness, highlight_ratio, shadow_ratio]
    """

    def __init__(self):
        super().__init__()
        # BT.601 RGB-to-grayscale weights
        self.register_buffer(
            "rgb_weights", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = (images * self.rgb_weights).sum(dim=1, keepdim=True)
        gray_flat = gray.view(gray.size(0), -1)

        mean_brightness = gray_flat.mean(dim=1)
        std_brightness = gray_flat.std(dim=1)
        # 220/255 ≈ 0.8627, 50/255 ≈ 0.1961
        highlight_ratio = (gray_flat > (220.0 / 255.0)).float().mean(dim=1)
        shadow_ratio = (gray_flat < (50.0 / 255.0)).float().mean(dim=1)

        return torch.stack(
            [mean_brightness, std_brightness, highlight_ratio, shadow_ratio],
            dim=1,
        )


class AestheticsFeatureExtractor(nn.Module):
    """Extract 7-D aesthetics features on GPU.

    Output (B, 7):
        [color_richness, saturation_mean, saturation_std, color_harmony,
         rms_contrast, edge_density, sharpness]
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "rgb_weights", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        )
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian_kernel", laplacian)

    def _rgb_to_hsv(self, images: torch.Tensor) -> torch.Tensor:
        """Convert RGB images in [0,1] to HSV with H in [0,180], S in [0,255]."""
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        max_val, max_idx = images.max(dim=1, keepdim=True)
        min_val = images.min(dim=1, keepdim=True)[0]
        diff = max_val - min_val

        s = torch.where(max_val > 0, diff / (max_val + 1e-8), torch.zeros_like(max_val))

        h = torch.zeros_like(max_val)
        mask_r = (max_idx == 0)
        h = torch.where(mask_r, 60.0 * ((g - b) / (diff + 1e-8)) % 360, h)
        mask_g = (max_idx == 1)
        h = torch.where(mask_g, 60.0 * ((b - r) / (diff + 1e-8)) + 120.0, h)
        mask_b = (max_idx == 2)
        h = torch.where(mask_b, 60.0 * ((r - g) / (diff + 1e-8)) + 240.0, h)
        h = h % 360.0
        h = h / 2.0  # OpenCV scale [0, 180]

        s = s * 255.0
        v = max_val * 255.0
        return torch.cat([h, s, v], dim=1)

    def _hue_entropy(self, h_flat: torch.Tensor) -> torch.Tensor:
        """Normalized entropy of a 12-bin hue histogram."""
        B = h_flat.size(0)
        num_bins = 12
        bin_width = 180.0 / num_bins
        bin_idx = (h_flat / bin_width).long().clamp(0, num_bins - 1)

        hist = torch.zeros(B, num_bins, device=h_flat.device, dtype=h_flat.dtype)
        hist.scatter_add_(1, bin_idx, torch.ones_like(h_flat))
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-7)

        entropy = -(hist * torch.log(hist + 1e-7)).sum(dim=1)
        max_entropy = math.log(num_bins)
        return (entropy / max_entropy).clamp(0.0, 1.0)

    def _edge_density(self, gray: torch.Tensor) -> torch.Tensor:
        """Edge pixel ratio via Sobel gradient magnitude (Canny approximation)."""
        gray_255 = gray * 255.0
        gx = F.conv2d(gray_255, self.sobel_x, padding=1)
        gy = F.conv2d(gray_255, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        edges = (mag > 100.0).float()
        return edges.view(edges.size(0), -1).mean(dim=1)

    def _laplacian_sharpness(self, gray: torch.Tensor) -> torch.Tensor:
        """Sharpness as variance of the Laplacian response."""
        gray_255 = gray * 255.0
        lap = F.conv2d(gray_255, self.laplacian_kernel, padding=1)
        variance = lap.view(lap.size(0), -1).var(dim=1)
        return (variance / 10000.0).clamp(0.0, 1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        hsv = self._rgb_to_hsv(images)
        h_channel = hsv[:, 0:1]
        s_channel = hsv[:, 1:2]
        gray = (images * self.rgb_weights).sum(dim=1, keepdim=True)

        h_flat = h_channel.view(B, -1)
        s_flat = s_channel.view(B, -1)
        gray_flat = gray.view(B, -1)

        # Color harmony (4 dims)
        color_richness = h_flat.std(dim=1) / 180.0
        saturation_mean = s_flat.mean(dim=1) / 255.0
        saturation_std = s_flat.std(dim=1) / 255.0
        color_harmony = self._hue_entropy(h_flat)

        # Contrast & sharpness (3 dims)
        rms_contrast = (gray_flat.std(dim=1) / 0.5).clamp(0.0, 1.0)
        edge_density = self._edge_density(gray)
        sharpness = self._laplacian_sharpness(gray)

        return torch.stack(
            [
                color_richness, saturation_mean, saturation_std, color_harmony,
                rms_contrast, edge_density, sharpness,
            ],
            dim=1,
        )


class MaterialFeatureExtractor(nn.Module):
    """Extract 10-D LBP texture histogram on GPU.

    Implements rotation-invariant uniform Local Binary Pattern with P=8, R=1.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "rgb_weights", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        )
        # 8 neighbors on the 3x3 ring (right, upper-right, ..., lower-right)
        self.register_buffer(
            "dy", torch.tensor([0, -1, -1, -1, 0, 1, 1, 1], dtype=torch.long)
        )
        self.register_buffer(
            "dx", torch.tensor([1, 1, 0, -1, -1, -1, 0, 1], dtype=torch.long)
        )

    def _classify_uniform(self, lbp_codes: torch.Tensor) -> torch.Tensor:
        """Map 8-bit LBP codes to 10 uniform-pattern bins."""
        P = 8
        transitions = torch.zeros_like(lbp_codes)
        for i in range(P):
            bit_i = (lbp_codes >> i) & 1
            bit_next = (lbp_codes >> ((i + 1) % P)) & 1
            transitions = transitions + (bit_i ^ bit_next)

        num_ones = torch.zeros_like(lbp_codes)
        for i in range(P):
            num_ones = num_ones + ((lbp_codes >> i) & 1)

        is_uniform = (transitions <= 2)
        return torch.where(is_uniform, num_ones, torch.full_like(num_ones, 9))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        gray = (images * self.rgb_weights).sum(dim=1) * 255.0
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode="constant", value=0)
        center = gray_pad[:, 1:H + 1, 1:W + 1]

        lbp_code = torch.zeros(B, H, W, device=images.device, dtype=torch.long)
        for i in range(8):
            ny = 1 + self.dy[i]
            nx = 1 + self.dx[i]
            neighbor = gray_pad[:, ny:ny + H, nx:nx + W]
            bit = (neighbor >= center).long()
            lbp_code = lbp_code + (bit << i)

        uniform_bin = self._classify_uniform(lbp_code.view(B, -1))

        n_bins = 10
        hist = torch.zeros(B, n_bins, device=images.device, dtype=images.dtype)
        hist.scatter_add_(
            1, uniform_bin, torch.ones_like(uniform_bin, dtype=images.dtype)
        )
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-7)
        return hist


# ============================================================
# Backbones (Swin V2-B + OpenCLIP)
# ============================================================

class SwinV2Backbone(nn.Module):
    """Swin Transformer V2-B visual backbone (output dim = 1024)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        else:
            self.backbone = swin_v2_b(weights=None)
        self.output_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class OpenCLIPTextEncoder(nn.Module):
    """OpenCLIP ViT-B-32-quickgelu text encoder (output dim = 512).

    Always frozen during training.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "laion400m_e32",
    ):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.output_dim = self.clip_model.text_projection.shape[-1]

    def forward(self, prompts: list, device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(device)
        return self.clip_model.encode_text(tokens)


# ============================================================
# Fusion / Projection / Head Modules
# ============================================================

class PhysicalVisualFusion(nn.Module):
    """Physical-Visual Outer Product Fusion (PV-OPF)."""

    def __init__(self, visual_dim: int, physical_dim: int, out_dim: int):
        super().__init__()
        self.compress_conv = nn.Conv1d(physical_dim, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, visual_feat: torch.Tensor, physical_feat: torch.Tensor) -> torch.Tensor:
        outer = physical_feat.unsqueeze(2) * visual_feat.unsqueeze(1)
        compressed = self.compress_conv(outer)
        out = (
            compressed.mean(dim=-1)
            if compressed.shape[-1] > 1
            else compressed.squeeze(-1)
        )
        return self.norm(out)


class CrossModalAttention(nn.Module):
    """Multi-head self-attention over the [image; text] sequence."""

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([image_feat.unsqueeze(1), text_feat.unsqueeze(1)], dim=1)
        attn_out, _ = self.multihead_attn(seq, seq, seq)
        attn_out = self.norm1(attn_out + seq)
        ffn_out = self.ffn(attn_out)
        out = self.norm2(ffn_out + attn_out)
        return out.mean(dim=1)


class DimensionAdapter(nn.Module):
    """Linear + LayerNorm + ReLU + Dropout dimension adapter."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class EnhancedMLP(nn.Module):
    """Progressive MLP head used as Quality Regression (QR) head."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PhysicalGateNetwork(nn.Module):
    """Sample-level scalar gate in (0, 1) used by Gated Residual Blending."""

    def __init__(self, visual_dim: int, physical_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(visual_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_proj: torch.Tensor, physical_feat: torch.Tensor) -> torch.Tensor:
        phys_proj = self.physical_proj(physical_feat)
        combined = torch.cat([visual_proj, phys_proj], dim=1)
        return self.gate_mlp(combined)


class BottleneckProjection(nn.Module):
    """Residual bottleneck projection: f + W_up ReLU(W_down f)."""

    def __init__(self, feature_dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        bottleneck_dim = feature_dim // bottleneck_ratio
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.projection(x)


# ============================================================
# SDIQA Model
# ============================================================

class SDIQAModel(nn.Module):
    """SDIQA quality assessment model.

    Visual backbone (Swin V2-B) is fully trainable.
    Text encoder (OpenCLIP) is frozen and used as a feature extractor.

    Args:
        unified_dim: Common feature dimensionality used by fusion modules.
        clip_model_name: OpenCLIP model identifier.
        clip_pretrained: OpenCLIP pretrained weights tag.
        visual_pretrained: Whether to load Swin V2-B ImageNet-1K weights.
        mlp_hidden_dims: Hidden dimensions of each QR head.
        mlp_dropout: Dropout for QR heads.
        adapter_dropout: Dropout for dimension adapters.
        num_attention_heads: Heads for cross-modal attention.
        light_feat_dim: Dimension of light physical feature (4).
        material_feat_dim: Dimension of material physical feature (10).
        aesthetics_feat_dim: Dimension of aesthetics physical feature (7).
        gate_hidden_dim: Hidden dim of physical gate networks.
        gate_dropout: Dropout for gate networks.
        bottleneck_ratio: Compression ratio for bottleneck projections.
    """

    def __init__(
        self,
        unified_dim: int = 1024,
        clip_model_name: str = "ViT-B-32-quickgelu",
        clip_pretrained: str = "laion400m_e32",
        visual_pretrained: bool = True,
        mlp_hidden_dims: List[int] = None,
        mlp_dropout: float = 0.2,
        adapter_dropout: float = 0.1,
        num_attention_heads: int = 8,
        light_feat_dim: int = 4,
        material_feat_dim: int = 10,
        aesthetics_feat_dim: int = 7,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
        bottleneck_ratio: int = 4,
    ):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256, 128]

        # ---- GPU physical feature extractors (no trainable params) ----
        self.light_extractor = LightFeatureExtractor()
        self.material_extractor = MaterialFeatureExtractor()
        self.aesthetics_extractor = AestheticsFeatureExtractor()

        # ---- Visual backbone (TRAINABLE) ----
        self.visual_backbone = SwinV2Backbone(pretrained=visual_pretrained)
        visual_dim = self.visual_backbone.output_dim

        # ---- Text encoder (FROZEN) ----
        self.text_encoder = OpenCLIPTextEncoder(
            model_name=clip_model_name, pretrained=clip_pretrained
        )
        text_dim = self.text_encoder.output_dim
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ---- Dimension adapters ----
        self.image_adapter = DimensionAdapter(visual_dim, unified_dim, dropout=adapter_dropout)
        self.text_adapter = DimensionAdapter(text_dim, unified_dim, dropout=adapter_dropout)

        # ---- Cross-modal attention (semantic stream) ----
        self.cross_modal_attention = CrossModalAttention(
            feature_dim=unified_dim, num_heads=num_attention_heads
        )

        # ---- Bottleneck projections (semantic stream) ----
        self.creativity_proj = BottleneckProjection(unified_dim, bottleneck_ratio=bottleneck_ratio)
        self.text_image_proj = BottleneckProjection(unified_dim, bottleneck_ratio=bottleneck_ratio)

        # ---- PGF module: PV-OPF + Visual Projection + Gated Residual Blending ----
        self.light_fusion = PhysicalVisualFusion(
            visual_dim=visual_dim, physical_dim=light_feat_dim, out_dim=unified_dim
        )
        self.material_fusion = PhysicalVisualFusion(
            visual_dim=visual_dim, physical_dim=material_feat_dim, out_dim=unified_dim
        )
        self.aesthetics_fusion = PhysicalVisualFusion(
            visual_dim=visual_dim, physical_dim=aesthetics_feat_dim, out_dim=unified_dim
        )
        self.physical_visual_projection = DimensionAdapter(
            visual_dim, unified_dim, dropout=adapter_dropout
        )

        self.light_gate = PhysicalGateNetwork(
            visual_dim=unified_dim, physical_dim=light_feat_dim,
            hidden_dim=gate_hidden_dim, dropout=gate_dropout,
        )
        self.material_gate = PhysicalGateNetwork(
            visual_dim=unified_dim, physical_dim=material_feat_dim,
            hidden_dim=gate_hidden_dim, dropout=gate_dropout,
        )
        self.aesthetics_gate = PhysicalGateNetwork(
            visual_dim=unified_dim, physical_dim=aesthetics_feat_dim,
            hidden_dim=gate_hidden_dim, dropout=gate_dropout,
        )

        # ---- Five quality regression heads ----
        self.mlp_text_image = EnhancedMLP(unified_dim, mlp_hidden_dims, dropout=mlp_dropout)
        self.mlp_material = EnhancedMLP(unified_dim, mlp_hidden_dims, dropout=mlp_dropout)
        self.mlp_light = EnhancedMLP(unified_dim, mlp_hidden_dims, dropout=mlp_dropout)
        self.mlp_creativity = EnhancedMLP(unified_dim, mlp_hidden_dims, dropout=mlp_dropout)
        self.mlp_aesthetics = EnhancedMLP(unified_dim, mlp_hidden_dims, dropout=mlp_dropout)

        # Cache for last batch's gate values (used by training-time regularization)
        self.last_gate_values: Dict[str, torch.Tensor] = {}

        logger.info("SDIQAModel initialized (visual trainable, text frozen)")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """Compute the five quality scores for each (image, prompt) pair.

        Args:
            images: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].
            prompts: List of B text prompts (str).

        Returns:
            Score tensor of shape (B, 5):
                [text_image, material, light, creativity, aesthetics]
        """
        device = images.device

        # 1. Visual features (trainable backbone)
        global_visual_feat = self.visual_backbone(images)

        # 2. Physical features (no grad, all on GPU)
        with torch.no_grad():
            light_physical_feat = self.light_extractor(images)
            material_physical_feat = self.material_extractor(images)
            aesthetics_physical_feat = self.aesthetics_extractor(images)

        # 3. Physical stream (PGF)
        visual_proj = self.physical_visual_projection(global_visual_feat)
        pv_opf_light = self.light_fusion(global_visual_feat, light_physical_feat)
        pv_opf_material = self.material_fusion(global_visual_feat, material_physical_feat)
        pv_opf_aesthetics = self.aesthetics_fusion(global_visual_feat, aesthetics_physical_feat)

        light_g = self.light_gate(visual_proj, light_physical_feat)
        material_g = self.material_gate(visual_proj, material_physical_feat)
        aesthetics_g = self.aesthetics_gate(visual_proj, aesthetics_physical_feat)

        fused_light_feat = (1 - light_g) * visual_proj + light_g * pv_opf_light
        fused_material_feat = (1 - material_g) * visual_proj + material_g * pv_opf_material
        fused_aesthetics_feat = (1 - aesthetics_g) * visual_proj + aesthetics_g * pv_opf_aesthetics

        self.last_gate_values = {
            "light": light_g, "material": material_g, "aesthetics": aesthetics_g,
        }

        # 4. Cross-modal branch (text encoder is frozen)
        with torch.no_grad():
            text_feat = self.text_encoder(prompts, device=device)
        text_feat = text_feat.detach()

        adapted_image_feat = self.image_adapter(global_visual_feat)
        adapted_text_feat = self.text_adapter(text_feat)
        fused_multimodal_feat = self.cross_modal_attention(
            adapted_image_feat, adapted_text_feat
        )

        # 5. Bottleneck projection for semantic stream
        creativity_feat = self.creativity_proj(fused_multimodal_feat)
        text_image_feat = self.text_image_proj(fused_multimodal_feat)

        # 6. Per-dimension scoring
        text_image_score = self.mlp_text_image(text_image_feat)
        material_score = self.mlp_material(fused_material_feat)
        light_score = self.mlp_light(fused_light_feat)
        creativity_score = self.mlp_creativity(creativity_feat)
        aesthetics_score = self.mlp_aesthetics(fused_aesthetics_feat)

        return torch.cat(
            [text_image_score, material_score, light_score,
             creativity_score, aesthetics_score],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict without the frozen OpenCLIP text encoder.

        The text encoder is always re-initialized from the official OpenCLIP
        weights, so excluding it keeps checkpoints compact.
        """
        return {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("text_encoder.")
        }

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return parameter counts for monitoring."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }


# ============================================================
# Checkpoint I/O
# ============================================================

def load_checkpoint(model: SDIQAModel, path: str,
                    map_location="cpu") -> Dict:
    """Load SDIQA weights saved by ``train.py`` into ``model``.

    Parameters of the frozen text encoder and constant buffers (e.g. Sobel
    kernels) may be missing since they are rebuilt at construction time; any
    other missing parameter or unexpected key raises an error.

    Returns:
        The raw checkpoint dictionary (contains ``epoch``, ``val_plcc``, ...).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt
    for key in ("model_state_dict", "model_trainable_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            break
    info = model.load_state_dict(state_dict, strict=False)

    param_names = {name for name, _ in model.named_parameters()}
    missing = [
        k for k in info.missing_keys
        if k in param_names and not k.startswith("text_encoder.")
    ]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:5]}, "
            f"unexpected={info.unexpected_keys[:5]}"
        )
    return ckpt
