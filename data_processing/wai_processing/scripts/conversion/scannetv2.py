# Copyright (c) Meta Platforms, Inc. and affiliates.
# Apache-2.0 License

import csv
import logging
import os
import struct
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from argconf import argconf_parse
from torchvision.io import decode_image
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,
    get_original_scene_names_slurm,
)

from mapanything.utils.wai.camera import gl2cv
from mapanything.utils.wai.core import store_data

logger = logging.getLogger(__name__)


COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}


class RGBDFrame():

    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.v2.imread(self.color_data)


class SensorData:

    def __init__(self, filename, verbose=False):
        self.version = 4
        self.verbose = verbose
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
            
            self.frames = [(i, f) for i, f in enumerate(self.frames) if not np.any(np.isinf(f.camera_to_world))]

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.verbose:
            print('exporting', len(self.frames)//frame_skip, 'depth frames to', output_path)

        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f][1].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            depth = depth / 1000.0
            depth = depth.astype(np.float32)
            store_data(os.path.join(output_path, str(self.frames[f][0]) + '.exr'), torch.from_numpy(depth), 'depth')

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if self.verbose:
            print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
        
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f][1].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(output_path, str(self.frames[f][0]) + '.jpg'), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.verbose:
            print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
        
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(self.frames[f][1].camera_to_world, os.path.join(output_path, str(self.frames[f][0]) + '.txt'))

    def export_intrinsics(self, output_path, frame_skip):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.verbose:
            print('exporting camera intrinsics to', output_path)
        
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))


def ensure_sens_exported(
    scan_dir: Path,
    export_dir: Path,
    overwrite: bool = False,
    frame_skip: int = 1,
) -> Dict[str, Path]:
    """
    Exports .sens into:
      export_dir/color, export_dir/depth, export_dir/pose, export_dir/intrinsic
    Returns dict of those dirs.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "color": export_dir / "color",
        "depth": export_dir / "depth",
        "pose": export_dir / "pose",
        "intrinsic": export_dir / "intrinsic",
    }
    for k, d in out.items():
        d.mkdir(parents=True, exist_ok=True)

    # Heuristic: if color already exists and not overwrite => skip export
    has_any_color = any(out["color"].glob("*.jpg")) or any(out["color"].glob("*.png"))
    if has_any_color and not overwrite:
        return out

    sens_files = list(scan_dir.glob("*.sens"))
    if len(sens_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .sens in {scan_dir}, found: {sens_files}"
        )
    sens_path = sens_files[0]

    sd = SensorData(sens_path, verbose=False)

    # Export color/depth/pose/intrinsics
    # These method names are stable in the official ScanNet SensReader.
    for fn_name, arg in [
        ("export_color_images", out["color"]),
        ("export_depth_images", out["depth"]),
        ("export_poses", out["pose"]),
        ("export_intrinsics", out["intrinsic"]),
    ]:
        if not hasattr(sd, fn_name):
            raise AttributeError(
                f"SensorData missing method {fn_name}. "
                "Please ensure you are using ScanNet's official SensReader/python."
            )
        fn = getattr(sd, fn_name)
        try:
            fn(str(arg), frame_skip=frame_skip)
        except TypeError:
            fn(arg, frame_skip=frame_skip)

    return out


# -----------------------------
# Helpers: label zip extraction
# -----------------------------
def ensure_zip_extracted(zip_path: Path, extract_dir: Path, overwrite: bool = False) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    # heuristic: if any png exists => assume extracted
    if any(extract_dir.rglob("*.png")) and not overwrite:
        return extract_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _frame_id_from_filename(p: Path) -> Optional[str]:
    return p.stem


def _index_frames(color_dir: Path) -> Dict[str, Path]:
    exts = ["*.jpg", "*.png", "*.jpeg"]
    files: List[Path] = []
    for e in exts:
        files.extend(color_dir.glob(e))
    out: Dict[str, Path] = {}
    for f in sorted(files):
        fid = _frame_id_from_filename(f)
        if fid is not None:
            out[fid] = f
    if not out:
        raise FileNotFoundError(f"No color frames found in {color_dir}")
    return out


def _find_matching_file(dir_path: Path, frame_id: str, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        cand = dir_path / pat.format(frame_id=frame_id)
        if cand.exists():
            return cand
    # fallback: search
    for f in dir_path.glob("*"):
        fid = _frame_id_from_filename(f)
        if fid == frame_id:
            return f
    return None


def _load_intrinsics(intr_dir: Path) -> Tuple[float, float, float, float]:
    """
    SensReader typically writes:
      intrinsic/intrinsic_color.txt
    and/or:
      intrinsic/intrinsic_depth.txt
    """
    candidates = [
        intr_dir / "intrinsic_color.txt",
        intr_dir / "intrinsic_depth.txt",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        raise FileNotFoundError(f"Missing intrinsics txt in {intr_dir}")

    K = np.loadtxt(path)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy


def _load_pose(pose_path: Path):
    T = np.loadtxt(pose_path).reshape(4, 4)
    # Convert OpenGL-like camera convention to OpenCV, consistent with scannetppv2 conversion style.
    # T = np.linalg.inv(T)
    # T, cmat = gl2cv(T, return_cmat=True)
    return T, None


def _safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)
    

# Standard ScanNet 20-class subset in NYU40-id space.
# We'll re-index them to training ids 1..20, and use 0 as ignore.
NYU40_20_IDS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]


def load_scannet_rawid_to_trainid_1to20(
    tsv_path: str | Path,
    *,
    nyu40_20_ids=None,
) -> Dict[int, int]:
    """
    Read scannet(v2)-labels.combined.tsv and return:
        raw_id (column 'id') -> train_id in [1..20]
    Any raw_id whose nyu40id is NOT in the 20-class subset will be unmapped (handled as ignore=0 later).

    Args:
        tsv_path: path to scannetv2-labels.combined.tsv (or scannet-labels.combined.tsv)
        nyu40_20_ids: iterable of NYU40 ids to keep; default is NYU40_20_IDS

    Returns:
        mapping dict {raw_id: train_id_1..20}
    """
    tsv_path = Path(tsv_path)
    if nyu40_20_ids is None:
        nyu40_20_ids = NYU40_20_IDS

    nyu40_20_ids_list = list(nyu40_20_ids)

    # Map NYU40 id -> contiguous train id 1..20
    nyu40_to_train = {nyu40_id: (i + 1) for i, nyu40_id in enumerate(nyu40_20_ids_list)}

    mapping: Dict[int, int] = {}

    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Empty/malformed TSV: {tsv_path}")

        # robust column lookup (case-insensitive)
        fn = {name.lower(): name for name in reader.fieldnames}
        if "id" not in fn or "nyu40id" not in fn:
            raise ValueError(
                f"TSV must contain columns 'id' and 'nyu40id'. Found: {reader.fieldnames}"
            )
        id_col = fn["id"]
        nyu40_col = fn["nyu40id"]

        for row in reader:
            try:
                raw_id = int(row[id_col])
                nyu40_id = int(row[nyu40_col])
            except Exception:
                continue

            if nyu40_id in nyu40_to_train:
                mapping[raw_id] = nyu40_to_train[nyu40_id]  # 1..20

    return mapping


def map_rawid_tensor_to_trainid_1to20(
    raw_id_img: torch.Tensor,
    rawid_to_trainid: Dict[int, int],
    *,
    ignore_label: int = 0,
    out_dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """
    Map (H,W) uint16 tensor of raw ids to:
      - mapped classes => 1..20
      - otherwise => 0 (ignore)

    Uses a 65536-sized LUT for speed.

    Args:
        raw_id_img: (H,W) torch.uint16
        rawid_to_trainid: dict {raw_id: train_id_1..20}
        ignore_label: default 0
        out_dtype: torch.uint8 by default (0..255 is enough)

    Returns:
        (H,W) tensor with values in {0..20}
    """
    if raw_id_img.dtype != torch.uint16:
        raise TypeError(f"raw_id_img must be torch.uint16, got {raw_id_img.dtype}")
    if raw_id_img.ndim != 2:
        raise ValueError(f"raw_id_img must be 2D (H,W), got shape {tuple(raw_id_img.shape)}")

    lut = torch.full((65536,), int(ignore_label), dtype=out_dtype, device=raw_id_img.device)

    for raw_id, train_id in rawid_to_trainid.items():
        if 0 <= raw_id < 65536:
            lut[raw_id] = int(train_id)

    return lut[raw_id_img.to(torch.int64)]


# -----------------------------
# Scene conversion
# -----------------------------
def convert_scene(cfg, scene_name: str, scannet_to_nyu_id_mapping: dict, **kwargs):
    """
    Converts one raw ScanNet scan (directory containing <scanId>.sens, mesh, zips...)
    into WAI format scene folder under cfg.root/scene_name.

    Adds frame modalities:
      - image
      - depth
      - semantic_label (if available)
      - instance_label (if available)

    Adds optional scene modalities (symlinks):
      - mesh files
      - aggregation/segs json
    """
    scan_dir = Path(cfg.original_root) / scene_name
    
    out_scene = Path(cfg.root) / scene_name
    out_scene.mkdir(parents=True, exist_ok=True)

    # 1) Export .sens (color/depth/pose/intrinsic) into a cache under original data (or configurable)
    export_dir = (
        Path(cfg.export_cache_root) / scene_name
        if getattr(cfg, "export_cache_root", None)
        else scan_dir / "_wai_export"
    )
    exported = ensure_sens_exported(
        scan_dir=scan_dir,
        export_dir=export_dir,
        overwrite=bool(getattr(cfg, "overwrite_export", False)),
        frame_skip=int(getattr(cfg, "frame_skip", 1)),
    )

    color_dir = exported["color"]
    depth_dir = exported["depth"]
    pose_dir = exported["pose"]
    intr_dir = exported["intrinsic"]

    # 2) Optional: extract 2D label zips (semantic + instance)
    # Prefer filtered zips if requested
    want_sem = bool(getattr(cfg, "enable_2d_semantic", True))
    want_ins = bool(getattr(cfg, "enable_2d_instance", True))
    prefer_filt = bool(getattr(cfg, "prefer_filtered_2d_labels", True))

    sem_dir = None
    ins_dir = None

    if want_sem:
        sem_zip = None
        if prefer_filt:
            cand = scan_dir / f"{scene_name}_2d-label-filt.zip"
            if cand.exists():
                sem_zip = cand
        if sem_zip is None:
            cand = scan_dir / f"{scene_name}_2d-label.zip"
            if cand.exists():
                sem_zip = cand

        if sem_zip is not None:
            sem_dir = ensure_zip_extracted(
                sem_zip,
                (export_dir / "_2d_label_filt") if "filt" in sem_zip.name else (export_dir / "_2d_label"),
                overwrite=bool(getattr(cfg, "overwrite_export", False)),
            )

    if want_ins:
        ins_zip = None
        if prefer_filt:
            cand = scan_dir / f"{scene_name}_2d-instance-filt.zip"
            if cand.exists():
                ins_zip = cand
        if ins_zip is None:
            cand = scan_dir / f"{scene_name}_2d-instance.zip"
            if cand.exists():
                ins_zip = cand

        if ins_zip is not None:
            ins_dir = ensure_zip_extracted(
                ins_zip,
                (export_dir / "_2d_instance_filt") if "filt" in ins_zip.name else (export_dir / "_2d_instance"),
                overwrite=bool(getattr(cfg, "overwrite_export", False)),
            )

    # 3) Index frames using exported color files
    color_frames = _index_frames(color_dir)

    # 4) Load intrinsics; load one image to get W/H (avoid extra deps; use imageio only if installed)
    fx, fy, cx, cy = _load_intrinsics(intr_dir)

    # derive width/height from first image
    w = h = None
    try:
        import imageio.v2 as imageio  # optional
        first = next(iter(color_frames.values()))
        img0 = imageio.imread(first)
        h, w = int(img0.shape[0]), int(img0.shape[1])
    except Exception:
        # If imageio not available, require user to provide w/h in config
        w = int(getattr(cfg, "w", 0))
        h = int(getattr(cfg, "h", 0))
        if w <= 0 or h <= 0:
            raise RuntimeError(
                "Cannot infer image size. Please install imageio or pass w/h in config."
            )

    # 5) Prepare output folder naming
    wai_dirs = {
        "images": "images",
        "depth": "depth",
        "semantic": "semantic_label",
        "instance": "instance_label",
        "mesh": "mesh",
        "annos": "annotations",
    }

    # 6) Build frames
    frames: List[Dict] = []
    missing_pose = 0
    missing_depth = 0
    missing_sem = 0
    missing_ins = 0

    for fid, img_path in sorted(color_frames.items()):
        # pose
        pose_path = _find_matching_file(
            pose_dir,
            fid,
            patterns=[
                "{frame_id}.txt",
                "frame-{frame_id}.pose.txt",
                "frame-{frame_id}.txt",
            ],
        )
        if pose_path is None or not pose_path.exists():
            missing_pose += 1
            continue

        # depth
        depth_path = _find_matching_file(
            depth_dir,
            fid,
            patterns=[
                "{frame_id}.png",
                "frame-{frame_id}.depth.png",
                "frame-{frame_id}.png",
            ],
        )
        if depth_path is None or not depth_path.exists():
            missing_depth += 1
            continue

        opencv_pose, gl2cv_cmat = _load_pose(pose_path)

        # semantic / instance label (optional)
        sem_path = None
        if sem_dir is not None:
            # files might be nested; search by frame id
            sem_path = next((p for p in sem_dir.rglob("*.png") if _frame_id_from_filename(p) == fid), None)
            if sem_path is None:
                missing_sem += 1

        ins_path = None
        if ins_dir is not None:
            ins_path = next((p for p in ins_dir.rglob("*.png") if _frame_id_from_filename(p) == fid), None)
            if ins_path is None:
                missing_ins += 1

        # Create symlinks into WAI scene directory
        # image
        target_img = out_scene / wai_dirs["images"] / f"{fid}.jpg"
        _safe_symlink(img_path, target_img)

        # depth
        target_depth = out_scene / wai_dirs["depth"] / f"{fid}.exr"
        _safe_symlink(depth_path, target_depth)

        frame = {
            "frame_name": fid,
            # Nerfstudio-compatible path
            "file_path": f"{wai_dirs['images']}/{fid}.jpg",
            "transform_matrix": opencv_pose.tolist(),
            "image": f"{wai_dirs['images']}/{fid}.jpg",
            "depth": f"{wai_dirs['depth']}/{fid}.exr",
        }

        if sem_path is not None:
            target_sem = out_scene / wai_dirs["semantic"] / f"{fid}.png"
            # _safe_symlink(sem_path, target_sem)
            sem = decode_image(sem_path)
            sem = sem[0]
            # breakpoint()
            sem = map_rawid_tensor_to_trainid_1to20(sem, scannet_to_nyu_id_mapping)
            sem = sem.to(torch.uint16)
            store_data(target_sem, sem, "labeled_image")
            frame["semantic_label"] = f"{wai_dirs['semantic']}/{fid}.png"

        if ins_path is not None:
            target_ins = out_scene / wai_dirs["instance"] / f"{fid}.png"
            # _safe_symlink(ins_path, target_ins)
            ins = decode_image(ins_path)
            ins = ins[0].to(torch.uint16)
            store_data(target_ins, ins, "labeled_image")
            frame["instance_label"] = f"{wai_dirs['instance']}/{fid}.png"

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No frames converted for {scene_name}. "
            f"missing_pose={missing_pose}, missing_depth={missing_depth}"
        )

    # 7) Scene-level modalities: mesh + 3D annotation jsons if present in raw folder
    scene_modalities: Dict[str, Dict] = {}

    # Mesh files listed in ScanNet README (common ones)
    for mesh_name in [
        f"{scene_name}_vh_clean.ply",
        f"{scene_name}_vh_clean_2.ply",
        f"{scene_name}_vh_clean_2.labels.ply",
    ]:
        src = scan_dir / mesh_name
        if src.exists():
            dst = out_scene / wai_dirs["mesh"] / mesh_name
            _safe_symlink(src, dst)

    # Annotation jsons
    for anno_name in [
        f"{scene_name}.aggregation.json",
        f"{scene_name}_vh_clean.aggregation.json",
        f"{scene_name}_vh_clean_2.0.010000.segs.json",
        f"{scene_name}_vh_clean.segs.json",
    ]:
        src = scan_dir / anno_name
        if src.exists():
            dst = out_scene / wai_dirs["annos"] / anno_name
            _safe_symlink(src, dst)

    # Populate scene_modalities records if desired
    # (These are optional; keep lightweight and transparent.)
    if (out_scene / wai_dirs["mesh"] / f"{scene_name}_vh_clean_2.ply").exists():
        scene_modalities["mesh_vh_clean_2"] = {
            "path": f"{wai_dirs['mesh']}/{scene_name}_vh_clean_2.ply",
            "format": "ply",
        }
    if (out_scene / wai_dirs["annos"] / f"{scene_name}.aggregation.json").exists():
        scene_modalities["aggregation"] = {
            "path": f"{wai_dirs['annos']}/{scene_name}.aggregation.json",
            "format": "json",
        }

    # 8) Assemble scene_meta (WAI format; store as scene_meta.json)
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": "scannet",
        "version": "0.1",

        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "shared_intrinsics": True,

        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,

        "frames": frames,

        "scene_modalities": scene_modalities,

        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
            # These two are optional; present only if zip existed + extracted
            "semantic_label": {
                "frame_key": "semantic_label",
                "format": "labeled_image",  # 16-bit png (ScanNet label ids)
            },
            "instance_label": {
                "frame_key": "instance_label",
                "format": "labeled_image",  # 8-bit png (instance ids)
            },
        },

        # "_applied_transform": gl2cv_cmat.tolist(),
        # "_applied_transforms": {"opengl2opencv": gl2cv_cmat.tolist()},
    }

    # 9) Write scene meta
    store_data(out_scene / "scene_meta.json", scene_meta, "scene_meta")

    logger.info(
        f"[{scene_name}] done. frames={len(frames)} "
        f"missing_pose={missing_pose} missing_depth={missing_depth} "
        f"missing_sem={missing_sem} missing_ins={missing_ins}"
    )


def main(cfg):
    scannet_to_nyu_id_mapping = load_scannet_rawid_to_trainid_1to20(
        Path(cfg.original_root).parent / "meta_data" / "scannetv2-labels.combined.tsv"
    )
    convert_scenes_wrapper(
        convert_scene,
        cfg,
        get_original_scene_names_func=get_original_scene_names_slurm,
        scannet_to_nyu_id_mapping=scannet_to_nyu_id_mapping,
    )


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/scannetv2.yaml")
    os.makedirs(cfg.root, exist_ok=True)

    main(cfg)
    # convert_scene(cfg, "scene0001_00")  # for quick test