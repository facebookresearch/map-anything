# The Challenges of Dynamic Point Map Reconstruction
## Overview
Reconstructing a dynamic point map from a single moving camera remains one of the most intricate challenges in 4D scene understanding. Unlike static structure-from-motion, which assumes a rigid world, dynamic reconstruction must disentangle camera motion from independently moving objects while maintaining both **spatial** and **temporal consistency**. The difficulty begins with **camera calibration** in dynamic environments—when parts of the scene are in motion, traditional feature correspondences used for intrinsic and extrinsic estimation often become unreliable, leading to drift or inconsistent scale.

At the heart of the problem lies the very definition of a *dynamic point map*. **Should points be persistently tracked across time—preserving their identities as they move**—or **should we instead distinguish between stationary and dynamic subsets, where stationary points maintain fixed coordinates through time while dynamic points emerge and vanish across frames (as in MonSt3R)?** Each interpretation imposes distinct assumptions on motion priors, correspondence strategies, and optimization objectives. 

## Defining the Dynamic Point Map as a 3D Point Tracking Problem Over Time
In recent research, point tracking is often defined as the process of tracking a set of points—either dense or sparse—initialized from the first frame. This tracking can occur either in the 2D RGB image space or in 3D geometric space (e.g., [DELTA: Dense Efficient Long-range 3D Tracking for Any Video](https://snap-research.github.io/DELTA/)). However, this formulation relies on a fixed set of trackable points, assuming that each point can be consistently followed forward through time. In practice, it is not feasible to track arbitrary points bidirectionally across time—that is, from any frame both forward and backward.

In theory, point tracking could serve as a foundation for dynamic point map reconstruction if any point from any frame could be tracked bidirectionally through time. However, this becomes an extremely difficult problem, as it implies that a video with dimensions $T \times W \times H$ would require defining $T \times W \times H$ individual tracklets, resulting in an overall trajectory tensor of shape $T \times T \times W \times H \times 3$.

## Motion Conveyed via Temporal Visibility of Points
In methods similar to MonST3R, point motion is not explicitly modeled. Instead, the dynamics of the dynamic point map are defined by the temporal visibility of points—that is, by their appearance and disappearance over time. A dynamic point cloud defined in this way can be represented as a tensor of shape $T\times W\times H \times 4$, where the additional channel encodes the visibility of each point.

## Comparing the Two Definitions of a Dynamic Point Cloud

The two formulations represent fundamentally different perspectives on motion.

- **Tracking-based definition:** Models continuous point trajectories through space and time, providing a physically grounded representation of motion. However, it struggles *when object topology changes*—or example, when a box opens or deforms—since point identities may no longer be preserved.

- **Visibility-based definition:** Represents motion implicitly through the temporal appearance and disappearance of points. This approach gracefully handles topology changes but sacrifices the notion of continuous point identity, making it less physically interpretable.