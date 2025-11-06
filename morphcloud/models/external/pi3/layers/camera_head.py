from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        )
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res


class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList(
            [deepcopy(ResConvBlock(output_dim, output_dim)) for _ in range(2)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(
            feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()
        )  ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out_t = self.fc_t(feat.float())  # [B,3]
            out_r = self.fc_rot(feat.float())  # [B,9]
            pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.0
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(
            torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2
        )
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1),
        )
        return r
