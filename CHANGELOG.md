# Changelog

All notable changes to MapAnything will be documented in this file.

## [1.1.0] - 2026-01-18

### Added

**Modular Architecture**
- Model factory interface (`model_factory`, `init_model_from_config`) for running different 3D reconstruction models through a unified API
- Support for external models: VGGT, DUSt3R, MASt3R, MUSt3R, Pi3-X, Pow3R, MoGe, AnyCalib, Depth Anything 3
- Unified output format across all model wrappers (`pts3d`, `pts3d_cam`, `ray_directions`, `cam_trans`, `cam_quats`, `conf`, etc.)
- Optional dependencies installation for external models via pip extras

**New Model Capabilities**
- DINO initialization support for Multi-View Transformer
- Relative Pose Loss + Absolute Pose Loss training objectives
- Memory efficient inference mode (`memory_efficient_inference=True`) enabling up to 2000 views on 140GB

**New Checkpoints**
- MapAnything V1.1 checkpoints on Hugging Face Hub
- V1 checkpoints preserved as `facebook/map-anything-v1` and `facebook/map-anything-apache-v1`

**Profiling**
- GPU memory usage and inference speed profiling script
- Comparison profiling against external models
- Visualization outputs (memory and speed plots)

**Datasets**
- AerialMegaDepth dataset integration to WAI format
- ScanNet++V2 rendering and config updates for WAI

**Demos and Tools**
- Demo script for running MapAnything on COLMAP outputs (`demo_inference_on_colmap_outputs.py`)
- Confidence slider in Gradio demo
- Pure local weight implementation

**Documentation**
- Copyright headers added to source files

### Fixed
- Inference with depth map as input (Issue #17)
- COLMAP demo handling of partial pose information
- Various misspellings across codebase

### Changed
- TA-WB download source migrated to Hugging Face Hub
- Removed MPSD downloader script (users must accept TOS on dataset website)
- Updated finetuning results documentation
- Cleaned up `hf_helpers.py`
- Updated installation instructions

## [1.0.0] - 2025-09-15

### Added
- Initial public release of MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- Complete codebase for inference, data processing, benchmarking, training, and ablations
- Two pre-trained model variants on Hugging Face Hub:
  - `facebook/map-anything` (CC-BY-NC 4.0 License) - Research & Academic Use
  - `facebook/map-anything-apache` (Apache 2.0 License) - Commercial Use
- Image-only inference support for metric 3D reconstruction from images
- Multi-modal inference support with flexible combinations of:
  - Images + Camera intrinsics
  - Images + Depth maps
  - Images + Camera poses
  - Any combination of the above inputs
- Interactive demos:
  - Online Hugging Face demo
  - Local Gradio demo with GUI interface
  - Rerun demo for interactive 3D visualization
- COLMAP & GSplat integration:
  - Direct export to COLMAP format
  - Bundle adjustment support
  - Gaussian Splatting compatibility
- Comprehensive data processing pipeline for 13 training datasets
- Complete training framework with:
  - Memory optimization support
  - All main model and ablation training scripts
  - Fine-tuning support for other geometry estimation models
- Benchmarking suite with:
  - Dense Up-to-N-View Reconstruction Benchmark
  - Single-View Image Calibration Benchmark
  - RobustMVD Benchmark
- Building blocks for the community:
  - UniCeption library for modular network components
  - WorldAI (WAI) unified data format for 3D/4D/Spatial AI
- Apache 2.0 licensed codebase for open-source development
- Complete documentation with installation instructions, API reference, and examples
