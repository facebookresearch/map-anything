# Visibility-Based Dynamics in 4D Point Representations
## Overview
> In methods similar to MonST3R, point motion is not explicitly modeled. Instead, the dynamics of the dynamic point map are defined by the temporal visibility of points—that is, by their appearance and disappearance over time. A dynamic point cloud defined in this way can be represented as a tensor of shape $T\times W\times H \times 4$, where the additional channel encodes the visibility of each point.

## Prior Work
In this work, our current model formulation still approaches dynamic point-map generation as a visibility-based point-cloud dynamics problem, rather than explicitly modeling point trajectories over time.

This choice is motivated by three factors:
1. Limitations in existing concurrent methods that we aim to address and improve upon.
2. The ability to directly leverage state-of-the-art SfM systems such as MapAnything and VGGT.
3. A simpler task formulation, which facilitates both model design and optimization.