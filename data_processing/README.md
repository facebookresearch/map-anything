# Data Processing for MapAnything

We use the WorldAI (WAI) format for our datasets and highly recommend using it for your own datasets. Stay tuned for our official public release! In the mean time, we provide an alpha version below to reproduce everything needed for MapAnything.

We also provide a HuggingFace dataset which contains the [pre-computed metadata](#pre-computed-metadata) for easy reproducibility of training & benchmarking.

## Public datasets converted to WAI format:

1. ✅ [Aria Synthetic Environments](https://www.projectaria.com/datasets/ase/)
2. ✅ [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
3. ✅ [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/)
4. ✅ [Dynamic Replica](https://dynamic-stereo.github.io/)
5. ✅ [ETH3D](https://www.eth3d.net/)
6. ✅ [Mapillary Planet Scale Depth & Reconstructions](https://www.mapillary.com/dataset/depth) (MPSD)
7. ✅ [MegaDepth (including Tanks & Temples)](https://www.cs.cornell.edu/projects/megadepth/)
8. ✅ [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
9. ✅ [Parallel Domain 4D](https://gcd.cs.columbia.edu/#datasets)
10. ✅ [SAIL-VOS 3D](https://sailvos.web.illinois.edu/_site/_site/index.html)
11. ✅ [ScanNet++ v2](https://kaldir.vc.in.tum.de/scannetpp/)
12. ✅ [Spring](https://spring-benchmark.org/)
13. ✅ [TartanAirV2 Wide Baseline](https://uniflowmatch.github.io/)
14. ✅ [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets)

## Download Instructions:

We provide instructions and scripts to download all the above datasets in [Data Download README](wai_processing/download_scripts/README.md).

## Installation Instructions:

### Base Setup

Create an environment with Python 3.12 (*this is different from the default one used for MapAnything*), for example with conda:

```bash
conda create -n wai_processing python=3.12 -y
conda activate wai_processing
cd <path to map-anything>
pip install --no-deps . # hydra conflicts with wai-processing requirements
cd data_processing/wai_processing/
```

*(Recommended option)* Full install of wai-processing in editable mode for convenient development:

```bash
pip install -e .[all]
```

or if you want a specific optional dependency, for example moge, you can do:
```bash
pip install -e .[moge] # install with moge support
```

### Additional MVSAnywhere Setup

In order to obtain mvsanywhere, please git clone the below repo and install it directly in your python environment like:

```bash
cd data_processing/wai_processing/
mkdir -p third_party && cd third_party && git clone https://github.com/arknapit/mvsanywhere.git && cd ..
mkdir -p third_party/mvsanywhere/checkpoints && cd third_party/mvsanywhere/checkpoints && wget https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt && cd ../../..
pip install -e .[mvsanywhere] # install with mvsanywhere support
```

MVSAnywhere stage is only needed for datasets without ground truth depth maps, i.e., only DL3DV-10K in our case.

## Running WAI Processing:

### Individual Stages

The different stages supported in the alpha release of WAI processing can be run using commands in the following format:

```bash
# Change directory to the MapAnything root folder
cd <path to map-anything>

# Run a conversion script (dataset specific)
python -m wai_processing.scripts.conversion.<dataset_name> \
          original_root=<original_dataset_path> \
          root=<wai_format_dataset_path>

# Run undistortion (modalities can be dataset specific)
python -m wai_processing.scripts.undistort \
          <specific_configs>.yaml \
          root=<dataset_path>

# Run rendering (relevant for datasets like ScanNet++V2)
python -m wai_processing.scripts.run_rendering \
          root=<dataset_path>

# Run covisibility
python -m wai_processing.scripts.covisibility \
          <specific_config>.yaml \
          root=<dataset_path>

# Run moge
python -m wai_processing.scripts.run_moge \
          root=<dataset_path>

# Run mvsanywhere
python -m wai_processing.scripts.run_mvsanywhere \
          root=<dataset_path>

# Get depth consistency confidence (for e.g., useful for mvsanywhere)
python -m wai_processing.scripts.depth_consistency_confidence \
          <specific_config>.yaml \
          root=<dataset_path>
```

Note that all these stages are not required for every dataset. Please refer to the launch configs shared below for the different required stages.

### Quick Start Example for ETH3D

Example commands to fully process the ETH3D dataset to WAI format as expected by MapAnything:

```bash
cd <path to map-anything>

# Run conversion
python -m wai_processing.scripts.conversion.eth3d \
          original_root="/fsx/xrtech/dryrun_tmp/eth3d_raw" \
          root="/fsx/xrtech/dryrun_tmp/eth3d"

# Run covisibility
python -m wai_processing.scripts.covisibility \
          data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
          root="/fsx/xrtech/dryrun_tmp/eth3d"

# Run moge
python -m wai_processing.scripts.run_moge \
          root="/fsx/xrtech/dryrun_tmp/eth3d" \
          batch_size=1 # MoGe stage doesn't support nested tensors
```

### Batch Processing using SLURM

In case you have access to a SLURM cluster you can use our slurm_launcher for easy batched processing of all the datasets.
From the SLURM job node, navigate to `<map-anything_repo_path>` and launch:

```bash
cd <path to map-anything>
python -m wai_processing.launch.slurm_stage \
  data_processing/wai_processing/configs/launch/<dataset_name>.yaml \
  conda_env=<name_of_your_conda_env> \
  stage=<stage according to the config> \
  launch_on_slurm=false
```

See `data_processing/wai_processing/configs/launch` for all the launch configs used to process the final datasets used for MapAnything.

## Pre-computed Metadata

To enable ease of reproducing MapAnything training \& benchmarking, we open source the pre-computed covisibility matrices for all the scenes. We also provide the train, validation and test split files containing respective scene names (generated using `data_processing/aggregate_scene_names.py`).

The data (115 GB in total) is hosted on [HuggingFace Datasets](https://huggingface.co/datasets/facebook/map-anything) and can be downloaded using the following command:

```python
from huggingface_hub import snapshot_download

# Download with parallel workers for faster download
snapshot_download(
    repo_id="facebook/map-anything",
    repo_type="dataset",
    local_dir="<root_dir>/map-anything-dataset",
    max_workers=24,  # Adjust based on your connection and system
)
```

## Visualizing WAI format Data

For ease of visualizing WAI format data after conversion, we provide a simple Rerun based visualizer:

```bash
cd <path to map-anything>
conda activate mapanything # need to use the default mapanything env
python3 data_processing/viz_data.py -h
```

## Adding Custom Datasets

Custom data can be easily be added to the WAI format using the WAI-processing tools. This subsequently enables easy training and benchmarking with MapAnything. You just need to add a conversion script in `data_processing/wai_processing/scripts/conversion` and corresponding configs in `data_processing/wai_processing/configs`.

## WAI Dataset Format

## Folder structure
We follow [Nerfstudio](https://docs.nerf.studio/quickstart/)’s folder structure and extend it by additional (optional) modalities.
The general folder and file structure is as follows:
```
<dataset name>
├── <first scene name>
│   ├── scene_meta.json   OR   scene_meta_distorted.json
│   │
│   ├── images   OR   images_distorted
│   │   ├── <frame_id_1>.[png|jpg]
│   │   :
│   │   └── <frame_id_n>.[png|jpg]
│   │
│   ├── [Optional] <depth> (GT depth, as specified in scene_meta.json)
│   │   ├── <frame_id_1>.exr
│   │   :
│   │   └── <frame_id_n>.exr
│   │
│   ├── [Optional] masks
│   │   ├── <frame_id_1>.png
│   │   :
│   │   └── <frame_id_n>.png
│   │
│   └── [Optional] Any extra modalities, as specified in scene_meta.json
:
└── <last scene name>
```

## `scene_meta` format

The `scene_meta.json` format is an extension of Nerfstudio's [transforms.json](https://docs.nerf.studio/quickstart/data_conventions.html).

The general structure is:
```json5
{
  "scene_name": "00dd871005", // Unique scene name
  "dataset_name": "scannetppv2", // Unique dataset name
  "version": "0.1", // WAI format version
  "last_modified": "2025-02-10T09:43:48.232022", // ISO datetime format
  "shared_intrinsics": true/false, // Same intrinsics for all cameras?
  // Camera model type [PINHOLE, OPENCV, OPENCV_FISHEYE]
  "camera_model": "PINHOLE",
  // Convention for cam2world extrinsics (must be "opencv")
  "camera_convention": "opencv",
  // <camera_coeff_name>: // camera coefficients like fl_x, cx, h,...
  // Per-frame intrinsics and extrinsics parameters
  "frames": <see below>,
  // Scene-level modalities with default mapping like gt_points3D -> pts3d.npy
  "scene_modalities": <see below>,
  // Frame-level modalities with default mapping like pred_depth -> metric3dv2
  "frame_modalities": <see below>,
  // Transform applied on original poses to get poses stored in `frames`
  "_applied_transform": <see below>,
  // All transforms to convert the original poses to poses stored in `frames`
  "_applied_transforms": <see below>,
}
```

Per-frame intrinsics can also be defined in the `frames` field, which is a list of dictionaries with the following structure:
```json5
{
  "frames": [
    {
      // Unique name to identify a frame
      "frame_name": "<frame_name>",
      // Relative path to frame, required for Nerfstudio compatibility
      "file_path": "<images>/<frame_name>.<ext>",
      // 4x4 flattened list of extrinsics in OpenCV format
      "transform_matrix": [[1, 0, 0, 0], ... [0, 0, 0, 1]],
      // Relative path to frame modality (optional)
      "<modality>_path": "<modality_path>/<frame_name>.<ext>",
      // Additional intrinsics for this frame (optional)
      "camera_model": "PINHOLE",
      // <camera_coeff_name>: // camera coefficients like fl_x, cx, h,...
    },
    ...
  ]
}
```

Example:
```json5
{
  "scene_name": "00dd871005", // unique scene name
  "dataset_name": "scannetppv2", // unique dataset name
  "version": "0.1",
  "last_modified":  "2025-02-10T09:43:48.232022",
  "camera_model": "PINHOLE", // camera model type [PINHOLE, OPENCV, OPENCV_FISHEYE]
  "camera_convention": "opencv", // camera convention used for cam2world extrinsics (different to Nerfstudio!)
  "fl_x": 1072.0, // focal length x
  "fl_y": 1068.0, // focal length y
  "cx": 1504.0, // principal point x
  "cy": 1000.0, // principal point y
  "w": 3008, // image width
  "h": 2000, // image height
  "frames": [
    {
      "frame_name": "000000",
      "file_path": "images/000000.png", // required by Nerfstudio
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ], // required by Nerfstudio
      "image": "images/000000.png", // same as file_path
      "metric3dv2_depth": "metric3dv2/v0/depth/000000.png",
      "metric3dv2_depth_conf": "metric3dv2/v0/depth_confidence/000000.exr",
      "fl_x": 1234, // specific focal length for this frame
      "w": 1000 // specific width for this frame
    },
    ...
  ],
  "scene_modalities": {
    "gt_pts3d": {
        "path": "global_pts3d.npy", //path to a scene_level point cloud
        "format": "numpy"
    }
  },
  "frame_modalities": {
    "pred_depth": {
        "frame_key": "metric3dv2_depth", //default mapping of pred_depth to frame modality
        "format": "depth"
    },
    "image": {
        "frame_key": "image", //default mapping of pred_depth to modality
        "format": "image",
    },
    "depth_confidence": {
        "frame_key": "metric3dv2_depth_conf",
        "format": "scalar"
    }
  },
  "_applied_transformation": [ // e.g. the transformation from opengl to opencv
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
  ],
  "_applied_transformations": {
    "opengl2opencv": [ // e.g. applied from OpenGL to OpenCV before
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
  ]}
}
```

## Citation

If you find our processed data, code, or repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{keetha2025mapanything,
  title={{MapAnything}: Universal Feed-Forward Metric {3D} Reconstruction},
  author={Nikhil Keetha and Norman Müller and Johannes Schönberger and Lorenzo Porzi and Yuchen Zhang and Tobias Fischer and Arno Knapitsch and Duncan Zauss and Ethan Weber and Nelson Antunes and Jonathon Luiten and Manuel Lopez-Antequera and Samuel Rota Bulò and Christian Richardt and Deva Ramanan and Sebastian Scherer and Peter Kontschieder},
  booktitle={arXiv},
  year={2025}
}
```
