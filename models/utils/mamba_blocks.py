import torch
import torch.nn as nn
from mamba_ssm import Mamba


class PatchEmbedding2d(nn.Module):
    """
    2D Image Patch Embedding module for Vision Mamba.
    Projects image patches to feature embeddings and adds 1D positional embeddings.
    """
    def __init__(self, img_size: tuple, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
    
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size[0] * self.grid_size[1], embed_dim))

    def forward(self, x):
        x = self.proj(x)
        B, C, gH, gW = x.shape
        x = x.view(B, C, gH * gW)
        x = torch.permute(x, (0, 2, 1))
        x = x + self.pos_embed
        return x


class MambaBlock(nn.Module):
    """
    Bidirectional 2D Vision Mamba block for spatial feature processing.
    """
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.mamba_forward = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_backward = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        y_forward = self.mamba_forward(x)
        x_flipped = torch.flip(x, dims=[1])
        y_backward = self.mamba_backward(x_flipped)
        y_backward = torch.flip(y_backward, dims=[1])

        out = y_forward + y_backward + residual
        return out


class VisionMambaEncoder(nn.Module):
    """
    2D Vision Mamba Encoder for extracting spatial feature maps from multi-view images.
    """
    def __init__(self, img_size: tuple, in_channels: int, L_blocks: int, Expand: int, D_hidden: int, N_ssm: int):
        super().__init__()
        self.expand = Expand if Expand else 2

        self.proj = PatchEmbedding2d(img_size=img_size, patch_size=16, in_channels=in_channels, embed_dim=D_hidden)
        self.grid_size = self.proj.grid_size

        self.encoder = nn.ModuleList([
            MambaBlock(D_hidden, d_state=N_ssm, d_conv=4, expand=self.expand) for _ in range(L_blocks)
        ])

    def forward(self, x, return_2d=True):
        B = x.shape[0]
        x = self.proj(x)
        for block in self.encoder:
            x = block(x)
        
        if return_2d:
            H_grid, W_grid = self.grid_size
            x = x.transpose(1, 2).contiguous().view(B, -1, H_grid, W_grid)
            
        return x


class TemporalMambaBlock(nn.Module):
    """
    Unidirectional Mamba block for 1D temporal recurrence with RMSNorm and residual connection.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, x, inference_params=None):
        residual = x
        x = self.norm(x)
        x = self.mamba(x, inference_params=inference_params)
        return x + residual


class TemporalMamba(nn.Module):
    """
    Applies L_blocks of 1D TemporalMambaBlocks along the temporal sequence dimension for each pixel in a BEV grid.
    Input shape:  [B, T, C, H, W]
    Output shape: [B, T, C, H, W]
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, L_blocks: int = 4):
        super().__init__()
        self.mamba_encoder = nn.ModuleList([
            TemporalMambaBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(L_blocks)
        ])

    def forward(self, x, inference_params=None):
        B, T, C, H, W = x.shape
        
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x_flat = x.view(B * H * W, T, C)
        
        for block in self.mamba_encoder:
            x_flat = block(x_flat, inference_params=inference_params)
        
        out = x_flat.view(B, H, W, T, C)
        out = out.permute(0, 3, 4, 1, 2).contiguous()
        return out


def test_vision_mamba_encoder():
    img_size = (800, 608)
    patch_size = 16
    D_hidden = 256
    N_ssm = 16
    Expand = 2
    L_blocks = 4
    batch_size = 2

    model = VisionMambaEncoder(
        img_size=img_size,
        in_channels=3,
        L_blocks=L_blocks,
        Expand=Expand,
        D_hidden=D_hidden,
        N_ssm=N_ssm
    ).cuda()

    dummy_img = torch.randn(batch_size, 3, img_size[0], img_size[1]).cuda()

    print(f"Input shape: {dummy_img.shape}")
    with torch.no_grad():
        out = model(dummy_img, return_2d=True)

    print(f"Output shape (2D): {out.shape}")
    H_grid = img_size[0] // patch_size
    W_grid = img_size[1] // patch_size
    expected_shape = (batch_size, D_hidden, H_grid, W_grid)
    assert out.shape == expected_shape, f"Unexpected shape! Expected {expected_shape}, got {out.shape}"
    print(f"VisionMambaEncoder test passed: {H_grid}x{W_grid} grid, {D_hidden} dimensions.")


def test_temporal_mamba():
    B = 2
    T = 5
    C_bev = 64
    H_bev, W_bev = 50, 50
    
    print("\nTesting TemporalMamba...")
    model = TemporalMamba(dim=C_bev).cuda()
    dummy_bev_seq = torch.randn(B, T, C_bev, H_bev, W_bev).cuda()
    
    print(f"Input temporal shape: {dummy_bev_seq.shape}")
    with torch.no_grad():
        out = model(dummy_bev_seq)
        
    print(f"Output temporal shape: {out.shape}")
    assert out.shape == dummy_bev_seq.shape, "Output shape mismatch in TemporalMamba!"
    print("TemporalMamba test passed successfully.")


if __name__ == "__main__":
    print("=== Running Mamba Blocks Tests ===")
    test_vision_mamba_encoder()
    test_temporal_mamba()