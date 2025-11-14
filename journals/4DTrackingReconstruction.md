# 3D Point Tracking Problem Over Time

## Overview
> In recent research, point tracking is often defined as the process of tracking a set of points—either dense or sparse—initialized from the first frame. This tracking can occur either in the 2D RGB image space or in 3D geometric space (e.g., [DELTA: Dense Efficient Long-range 3D Tracking for Any Video](https://snap-research.github.io/DELTA/)). However, this formulation relies on a fixed set of trackable points, assuming that each point can be consistently followed forward through time. In practice, it is not feasible to track arbitrary points bidirectionally across time—that is, from any frame both forward and backward. In theory, point tracking could serve as a foundation for dynamic point map reconstruction if any point from any frame could be tracked bidirectionally through time. However, this becomes an extremely difficult problem, as it implies that a video with dimensions $T \times W \times H$ would require defining $T \times W \times H$ individual tracklets, resulting in an overall trajectory tensor of shape $T \times T \times W \times H \times 3$.

## The Training Criterions
Tracking $T \times W \times H$ pixels across a video sequence is computationally expensive, and there are no ground-truth labels available for these trajectories. Appearance supervision (photoemetric criterion) is needed for training, and 3D point tracking further requires a reprojection loss for consistency.

Let the point tracklet tensor be denoted as $\bar{\bm{K}} \in \mathbb{R}^{T\times T \times W \times H \times 3}$. The slice $\bar{\bm{K}}[\tilde{t}, \dots]$ corresoponds to the spatial locations of all pixels (from all frames) at the time step $\tilde{t}$. The sub-tensor $\bar{\bm{K}}[\tilde{t}, \hat{t}]$ represents the spatial locations of all pixels from frame $\hat{t}$ at the time step $\tilde{t}$.

Let the video frames be $$\bar{\bm{I}} \in \mathbb{R}^{T \times W \times H \times 3}$$By combining $\bar{\bm{K}}$ and $\bar{\bm{I}}$, we construct a large dynamic point map consisting of $T \times W \times H$ points tracked over $T$ time steps. The appearance of each point is given by $\bar{\bm{I}}$, while its spatial location over time is provided by $\bar{\bm{K}}$.

Let $\mathcal{R}$ denote a rendering function that takes as input a static point cloud and a camera pose $\pi$, and produces a rendered image of the point cloud from that camera viewpoint. The tensor $\bar{\bm{K}}[t, \dots]$. and $\bar{\bm{I}}$ defines a static point cloud, specificially it is the static scene cloud at time $t$ with all pixel from all the frames visible.

Let $\Pi$ denote the camera-pose sequence associated with $\bar{\bm{I}}$. Define $$\mathcal{R}(\bar{\bm{K}}, \bar{\bm{I}}, \Pi, t)$$ as the rendering of the scene at time $t$, which uses the point locations from the slice $\bar{\bm{K}}[t,\dots]$, the video appearance $\bar{\bm{I}}$ and the camera poses $\Pi_t$.

We also define $\mathcal{R}(\bar{\bm{K}}_{\neg{t}}, \bar{\bm{I}}_{\neg{t}}, \Pi, t)$ as the rendering of the scene at timestep $t$ with the tracklets from timestep $t$ removed. Here, $$\bar{\bm{K}}_{\neg{t}}=\bar{\bm{K}}[t, 1:t-1 \cup t+1:T]$$ denotes the set of point trajectories excluding those originating from frame $t$, and $\bar{\bm{I}}_{\neg{t}}$ likewise excludes the appearance of frame $t$.

With this formulation, we can introduce a photometric training objective that encourages the reconstructed dynamic point map to faithfully reproduce the observed video frames. Specifically, for each timestep $t$, we can compare the ground-truth frame $\bar{\bm{I}}_t$ to a leave-$t$-out rendering of the scene: $$\text{MSE}(\bar{\bm{I}}_t, \mathcal{R}(\bar{\bm{K}}_{\neg{t}}, \bar{\bm{I}}_{\neg{t}}, \Pi, t)).$$

This “leave-t-out” design prevents the model from trivially copying pixel colors from frame $t$ and instead forces the dynamic point map to reconstruct the frame from its tracked 3D structure and the camera trajectory. Minimizing this MSE thus encourages:
1. **Spatial consistency:** point trajectories must place 3D points at correct image-plane projections,

2. **Temporal consistency:** points from neighboring frames must agree on appearance and motion, and

3. **Cross-frame predictive power:** the representation must be able to explain frame $t$ using information propagated from other timesteps.

Aggregating the loss over all $t$ provides a global photometric objective for training the dynamic point map.