import torch


@torch.no_grad()
def pca_visualize_features(features):
    """
    Args:
        features: Tensor of shape (*, C).
    """
    shape = features.shape
    C = shape[-1]
    features = features.reshape(-1, C)  # (N, C)

    mean = features.mean(dim=0, keepdim=True)  # (1, C)
    X = features - mean  # (N, C)

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    pcs = Vh[:3].T

    proj_all = X @ pcs  # (N, 3)
    cmin = proj_all.min(dim=0).values
    cmax = proj_all.max(dim=0).values
    crng = (cmax - cmin).clamp(min=1e-6)

    rgb = (proj_all - cmin) / crng  # (N, 3)
    rgb = rgb.reshape(*shape[:-1], 3)  # (*, 3)

    return rgb