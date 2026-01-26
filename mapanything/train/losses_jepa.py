import warnings

import torch
import torch.nn as nn

# import lejepa


class OverfittingLoss(nn.Module):
    """
    Test loss to check if JEPA can overfit original features.
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch, x):
        original_feats, adapter_feats, _ = x

        original_feats = original_feats[-1]
        adapter_feats = adapter_feats[-1]

        original_feats = torch.stack(original_feats.features, dim=0)
        adapter_feats = torch.stack(adapter_feats.features, dim=0)

        loss = self.mse_loss(original_feats, adapter_feats)

        return loss, {'of_loss': loss}
    

class L2Loss(nn.Module):
    """
    Simple L2 loss for JEPA training.
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch, contexts, targets):

        # Seperate features and masks
        _, preds, target_mask = contexts
        targets, _, _ = targets

        targets = targets[-1]

        num_views = len(preds.features)
        batch_size, dim, _, _ = preds.features[0].shape

        preds = torch.stack(preds.features, dim=0)  # (V, B, C, H, W)
        targets = torch.stack(targets.features, dim=0)

        preds = preds[target_mask]  # (M, B, C, H, W)
        targets = targets[target_mask]

        inv_loss = self.mse_loss(preds, targets.detach())

        return inv_loss, {'pred_loss': inv_loss}


class LeJEPALoss(nn.Module):
    """
    Loss function for LeJEPA. Abandoned.

    Combines invariant prediction loss with SIGReg loss.
    This function does not yield reasonable results.
    """

    def __init__(self, lamb=0.02, t_max=5, n_points=17, n_slices=1024):
        super(LeJEPALoss, self).__init__()
        self.lamb = lamb
        self.mse_loss = nn.MSELoss()
        univariate_test = EppsPulley(t_max=t_max, n_points=n_points)
        self.sigreg = SlicingUnivariateTest(
            univariate_test=univariate_test, 
            num_slices=n_slices
        )

    def forward(self, batch, contexts, targets):
        """
        Compute the JEPA loss.

        Args:
            batch (list): List of view dictionaries containing input data.
            contexts (list): List of context predictions from the model.
            targets (list): List of target predictions from the model.

        Returns:
            torch.Tensor: Computed loss value.
        """
        # Seperate features and masks
        contexts, target_mask = contexts
        targets, _ = targets

        # Features contain outputs from multiple layers, we only use the last layer's
        contexts = contexts[-1]
        targets = targets[-1]

        num_views = len(contexts.features)
        batch_size, dim, _, _ = contexts.features[0].shape

        contexts = torch.stack(contexts.features, dim=0)  # (V, B, C, H, W)
        targets = torch.stack(targets.features, dim=0)

        contexts = contexts[target_mask]  # (M, B, C, H, W)
        targets = targets[target_mask]

        inv_loss = self.mse_loss(contexts, targets)

        # return inv_loss, {'pred_loss': inv_loss}

        contexts = contexts.permute(1, 2, 0, 3, 4).flatten(2)  # (B, C, M * H * W)
        targets = targets.permute(1, 2, 0, 3, 4).flatten(2)

        contexts = contexts.transpose(-1, -2)  # (B, M * H * W, C)
        targets = targets.transpose(-1, -2)

        sigreg_loss = self.sigreg(torch.cat([contexts, targets], dim=1))
        
        details = {'pred_loss': inv_loss, 'sigreg_loss': sigreg_loss}
        total_loss = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss

        return total_loss, details


class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s
    

class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point for linear spacing methods.
            Only used for 'trapezoid' and 'simpson' integration. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd for
            'simpson' integration. For 'gauss-hermite', this determines the number
            of positive nodes. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule with linear spacing over [0, t_max]
            Default: 'trapezoid'.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - For 'gauss-hermite', nodes and weights are adapted from the standard
          Gauss-Hermite quadrature to integrate against exp(-t²).
        - Supports distributed training via all_reduce operations.

    Example:
        >>> test = EppsPulley(t_max=5.0, n_points=21, integration='simpson')
        >>> samples = torch.randn(1000)  # Standard normal samples
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
    """

    def __init__(
        self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"
    ):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points
        # Precompute phi

        # Linearly spaced positive points (including 0)
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half-weight at t=0
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        # Compute cos/sin only for t >= 0
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across batch
        cos_mean = cos_vals.mean(-3)  # (*, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, n_points)

        # Compute error (symmetry already in weights)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integration
        return (err @ self.weights) * N


class SlicingUnivariateTest(torch.nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    This module extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions (slices) and aggregating univariate test statistics
    across all projections. The approach is based on the sliced method for comparing
    high-dimensional distributions.
    The test projects multivariate samples x ∈ ℝᴰ onto random unit vectors:
        x_projected = x @ A
    where A ∈ ℝᴰˣᴷ contains K normalized random direction vectors. A univariate
    test is then applied to each of the K projected samples, and results are aggregated.
    Args:
        univariate_test (torch.nn.Module): A univariate test module that accepts
            (*, N, K) tensors and returns (*, K) test statistics, where N is the
            number of samples and K is the number of slices.
        num_slices (int): Number of random 1D projections (slices) to use. More
            slices increase test power but add computational cost.
        reduction (str, optional): How to aggregate statistics across slices:
            - 'mean': Return the average statistic across all slices
            - 'sum': Return the sum of statistics across all slices
            - None: Return individual statistics for each slice (*, num_slices)
            Default: 'mean'.
        sampler (str, optional): Random sampling method for projection directions:
            - 'gaussian': Sample from standard normal distribution (Gaussian projections)
            Default: 'gaussian'.
        clip_value (float, optional): Minimum threshold for test statistics. Values
            below this threshold are clipped to zero. Useful for reducing noise from
            negligible deviations. Default: None (no clipping).
    Attributes:
        global_step (torch.Tensor): Counter for deterministic random seed generation,
            synchronized across distributed processes to ensure consistent projections.
    Notes:
        - Projection directions are normalized to unit vectors (L2 norm = 1).
        - In distributed training, the random seed is synchronized across all ranks
          using all_reduce to ensure identical projections on all devices.
        - The generator is cached and reused across forward passes for efficiency.
        - The global step counter increments after each forward pass to ensure
          different random projections in successive calls.
    Shape:
        - Input: (*, N, D) where * is any number of batch dimensions, N is the
          number of samples, and D is the feature dimension.
        - Output:
            - Scalar if reduction='mean' or 'sum'
            - (*, num_slices) if reduction=None
    Example:
        >>> from your_module import FastEppsPulley, SlicingUnivariateTest
        >>>
        >>> # Create univariate test
        >>> univariate_test = FastEppsPulley(t_max=5.0, n_points=21)
        >>>
        >>> # Wrap with slicing for multivariate testing
        >>> test = SlicingUnivariateTest(
        ...     univariate_test=univariate_test,
        ...     num_slices=100,
        ...     reduction='mean',
        ...     sampler='gaussian',
        ...     clip_value=0.01
        ... )
        >>>
        >>> # Test multivariate samples
        >>> samples = torch.randn(1000, 50)  # 1000 samples, 50 dimensions
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
        >>>
        >>> # Batch processing
        >>> batch_samples = torch.randn(32, 1000, 50)  # 32 batches
        >>> batch_stats = test(batch_samples)  # Returns scalar (averaged over slices)
    References:
        - Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012). Wasserstein
          barycenter and its application to texture mixing. In Scale Space and
          Variational Methods in Computer Vision (pp. 435-446).
        - Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and
          Radon Wasserstein barycenters of measures. Journal of Mathematical
          Imaging and Vision, 51(1), 22-45.
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value

    def _get_generator(self, device, seed):
        """Get or create generator for given device and seed."""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        Apply sliced univariate test to multivariate samples.
        Args:
            x (torch.Tensor): Input samples of shape (*, N, D) where * represents
                any number of batch dimensions, N is the number of samples, and
                D is the feature dimension.
        Returns:
            torch.Tensor: Aggregated test statistic(s).
                - Scalar tensor if reduction='mean' or 'sum'
                - Shape (*, num_slices) if reduction=None
        """
        with torch.no_grad():
            dev = dict(device=x.device)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev)
            A /= A.norm(p=2, dim=0)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats