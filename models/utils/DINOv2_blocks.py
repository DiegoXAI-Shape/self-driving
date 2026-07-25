import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class DINOv2EncoderLoRA(nn.Module):
    """
    DINOv2 Small (dinov2_vits14) feature extractor with traditional LoRA adaptation.
    """
    def __init__(self, model_name: str = "dinov2_vits14", patch_size: int = 14, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = 384  # dinov2_vits14 embedding dimension

        # Load pre-trained DINOv2 backbone from PyTorch Hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)

        # Freeze original backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Configure traditional LoRA for Multi-Head Self-Attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.05,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B_total, 3, H, W] (where B_total = B * S * N)
        Returns:
            feature_map_2d: Tensor of shape [B_total, 384, H_grid, W_grid]
        """
        B_total, C, H, W = x.shape

        # Ensure height and width are divisible by DINOv2 patch size (14)
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        H_grid = H_padded // self.patch_size
        W_grid = W_padded // self.patch_size

        # Extract patch features using DINOv2 forward_features
        features_dict = self.backbone.forward_features(x)
        patch_features = features_dict["x_norm_patchtokens"]  # Shape: [B_total, H_grid * W_grid, 384]

        # Reshape flat patch tokens to 2D spatial feature map [B_total, 384, H_grid, W_grid]
        feature_map_2d = patch_features.permute(0, 2, 1).contiguous().view(B_total, self.embed_dim, H_grid, W_grid)
        return feature_map_2d


def test_dinov2_encoder():
    print("==================================================================")
    print("   Testing DINOv2 Encoder with Traditional LoRA Adaptation        ")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = DINOv2EncoderLoRA(model_name="dinov2_vits14", lora_r=8, lora_alpha=16).to(device)

    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    B_total = 8  # 1 sample * 8 cameras
    dummy_input = torch.randn(B_total, 3, 308, 406).to(device)
    print(f"Input shape:  {dummy_input.shape}")

    encoder.eval()
    with torch.no_grad():
        out_features = encoder(dummy_input)

    print(f"Output 2D feature map shape: {out_features.shape}")
    expected_grid = (308 // 14, 406 // 14)  # (22, 29)
    assert out_features.shape == (B_total, 384, expected_grid[0], expected_grid[1]), \
        f"Shape mismatch! Expected {(B_total, 384, expected_grid[0], expected_grid[1])}, got {out_features.shape}"

    print(f"[OK] DINOv2 + LoRA test passed! Grid dimensions: {expected_grid[0]}x{expected_grid[1]}, Channels: 384.")


if __name__ == "__main__":
    test_dinov2_encoder()
