# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from argconf import argconf_parse
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
    get_original_scene_names,  # noqa: F401, Needed for launch_slurm.py
)

from morphcloud.utils.wai.camera import CAMERA_KEYS, gl2cv
from morphcloud.utils.wai.core import load_data, store_data
from morphcloud.utils.wai.semantics import INVALID_ID, load_semantic_color_mapping

logger = logging.getLogger(__name__)


def get_semantic_class_mapping(cfg):
    # map ScanNetv2 semantic labels with official mapping (str->int)
    source_labels_path = (
        Path(cfg.original_root).parent / "metadata" / "semantic_classes.txt"
    )
    with open(source_labels_path) as f:
        semantic_classes = f.read().splitlines()  # labels
    source_mapping_path = (
        # original root points to data --> original_root/../metadata is needed
        Path(cfg.original_root).parent
        / "metadata"
        / "semantic_benchmark"
        / "map_benchmark.csv"
    )
    original_str2str_mapping = pd.read_csv(source_mapping_path)
    # map class names to unified class names (e.g. books -> book)
    semantic_class_str2str_mapping = map_scannetv2_semantic_class(
        original_str2str_mapping, "semantic"
    )
    # map class name to index 0..N in the same order (e.g. book -> 5)
    semantic_class_str2id_mapping = {
        label: ndx for (ndx, label) in enumerate(semantic_classes)
    }
    semantic_class_mappings = {
        "str2id": semantic_class_str2id_mapping,
        "str2str": semantic_class_str2str_mapping,
    }
    return semantic_class_mappings


def map_scannetv2_semantic_class(mapping: pd.DataFrame, method: str):
    """
    Adapted from: https://github.com/scannetpp/scannetpp/blob/main/semantic/transforms/mesh.py
    Maps the semantic classes strings to a filtered set of classes strings.
    """
    if method == "semantic":
        map_key = "semantic_map_to"
    elif method == "instance":
        map_key = "instance_map_to"
    else:
        raise ValueError(f"Invalid method: {method}.")

    new_classes = []
    # create a dict with classes to be mapped
    # classes that dont have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> dont add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> dont use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
                    # x->x explicitly in mapping - allow this
                    if (class_name == map_target) and class_name not in new_classes:
                        new_classes.append(class_name)
                    # x->y but y not in list
                    if map_target not in new_classes:
                        new_classes.append(map_target)
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in new_classes:
                new_classes.append(class_name)
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


def map_semantic_class_to_index(
    source_anno,
    semantic_class_mappings,
    scannet_invalid_id=-100,
):
    """
    Adapted from: https://github.com/scannetpp/scannetpp/blob/main/semantic/transforms/mesh.py
    Maps the semantic class of the annotations (str) to the semantic label id (int) in the mapping.
    """

    # keep track of used semantic classes (to be stored in scene mappings)
    scene_semantic_class_mapping = {
        str(INVALID_ID): {
            "original_id": str(scannet_invalid_id),
            "original_name": "invalid",
            "mapped_name": "invalid",
        }
    }

    for ndx, anno in enumerate(source_anno["segGroups"]):
        original_semantic_class_name = anno["label"]

        # store original label
        source_anno["segGroups"][ndx]["label_orig"] = original_semantic_class_name

        # remap labels, e.g. books->book
        mapped_semantic_class_name = semantic_class_mappings["str2str"].get(
            original_semantic_class_name, None
        )
        # in case label is remapped - put the new label into the anno dict
        source_anno["segGroups"][ndx]["label"] = mapped_semantic_class_name

        # get label id (NOTE: adding a +1 offset to account for INVALID_ID=0)
        scannet_semantic_class_id = semantic_class_mappings["str2id"].get(
            mapped_semantic_class_name, scannet_invalid_id
        )
        if scannet_semantic_class_id == scannet_invalid_id:
            mapped_semantic_class_id = INVALID_ID
        else:
            mapped_semantic_class_id = scannet_semantic_class_id + 1
        source_anno["segGroups"][ndx]["label_ndx"] = mapped_semantic_class_id

        # track used labels
        if mapped_semantic_class_id != INVALID_ID:
            scene_semantic_class_mapping[str(mapped_semantic_class_id)] = {
                "original_id": str(scannet_semantic_class_id),
                "original_name": original_semantic_class_name,
                "mapped_name": mapped_semantic_class_name,
            }

    return source_anno, scene_semantic_class_mapping


def map_semantics_on_vertices(
    source_segments,
    source_anno,
    max_gt=3,
):
    """
    Adapted from: https://github.com/scannetpp/scannetpp/blob/main/semantic/transforms/mesh.py
    Assigns a semantic class and instance label to each vertex of the mesh.
    """

    seg_indices = np.array(source_segments["segIndices"], dtype=np.uint32)
    num_verts = len(seg_indices)

    # first store multilabels into array
    # if using single label, keep the label of the smallest instance for each vertex
    # else, keep everything

    # semantic multilabels
    multilabels = np.full((num_verts, max_gt), INVALID_ID, dtype=np.int16)
    # how many labels are used per vertex? initially 0
    # increment each time a new label is added
    # 0, 1, 2 eg. if max_gt is 3
    labels_used = np.zeros(num_verts, dtype=np.int16)
    # keep track of the size of the instance (#vertices) assigned to each vertex
    # later, keep the label of the smallest instance for multilabeled vertices
    # store inf initially so that we can pick the smallest instance
    instance_size = np.full((num_verts, max_gt), np.inf, dtype=np.float32)

    # all instance labels, including multilabels
    instance_multilabels = None
    # the final instance labels
    vertex_instance = None

    # new instance IDs from 0..N
    instance_multilabels = np.ones((num_verts, max_gt), dtype=np.int16) * INVALID_ID

    for instance_ndx, instance in enumerate(source_anno["segGroups"]):
        if instance["label_ndx"] == INVALID_ID:
            continue
        # get all the vertices with segment index in this instance
        # and max number of labels not yet applied
        inst_mask = np.isin(seg_indices, instance["segments"]) & (labels_used < max_gt)

        num_vertices = inst_mask.sum()
        if num_vertices == 0:
            continue

        # get the position to add the label - 0, 1, 2
        new_label_position = labels_used[inst_mask]
        multilabels[inst_mask, new_label_position] = instance["label_ndx"]

        # add instance label only for instance classes
        instance_multilabels[inst_mask, new_label_position] = instance_ndx

        # store number of vertices in this instance
        instance_size[inst_mask, new_label_position] = num_vertices
        labels_used[inst_mask] += 1

    # keep only the smallest instance for each vertex
    vertex_semantic_class = multilabels[:, 0]
    # vertices which have multiple labels
    has_multilabel = labels_used > 1
    # get the label of the smallest instance for multilabeled vertices
    smallest_instance_ndx = np.argmin(instance_size[has_multilabel], axis=1)
    vertex_semantic_class[has_multilabel] = multilabels[
        has_multilabel, smallest_instance_ndx
    ]

    # pick the 1st label for everything
    vertex_instance = instance_multilabels[:, 0]
    # pick the label of the smallest instance for multilabeled vertices
    vertex_instance[has_multilabel] = instance_multilabels[
        has_multilabel, smallest_instance_ndx
    ]

    return vertex_semantic_class, vertex_instance


def convert_scene(
    cfg,
    scene_name,
    modality,
    scannet_semantic_class_mappings,
    semantic_color_mapping,
):
    dataset_name = cfg.get("dataset_name", "scannetppv2")
    version = cfg.get("version", "0.2")

    org_scene_root = Path(cfg.original_root) / scene_name
    org2wai = {
        "resized_images": "images_distorted",
        "resized_anon_masks": "anon_masks_distorted",
    }

    logger.info(f"{scene_name}: Processing")

    out_path = Path(cfg.root) / scene_name

    ## Read all test scenes to exclude the test frames for those
    with open(cfg.test_split_fn, "r", encoding="utf-8") as f:
        test_scene_names = [line.strip() for line in f.readlines()]

    transforms_fn = Path(org_scene_root) / modality / "nerfstudio" / "transforms.json"
    meta = load_data(transforms_fn)
    if scene_name not in test_scene_names:
        frames = meta["frames"] + meta["test_frames"]
    else:
        frames = meta["frames"]

    frames.sort(key=lambda x: x["file_path"])
    # mapping used to look up frame based on frame_name
    test_frames = [f["file_path"] for f in meta["test_frames"]]
    train_frame_names = []
    eval_frame_names = []
    wai_frames = []

    image_out_path = out_path / org2wai["resized_images"]
    image_out_path.mkdir(parents=True, exist_ok=True)

    has_mask = Path(org_scene_root, modality, "resized_anon_masks").exists()
    if has_mask:
        anon_mask_out_path = out_path / org2wai["resized_anon_masks"]
        anon_mask_out_path.mkdir()

    for frame in frames:
        frame_name = Path(frame["file_path"]).stem
        wai_frame = {"frame_name": frame_name}
        org_transform_matrix = np.array(frame["transform_matrix"]).astype(np.float32)
        opencv_pose, gl2cv_cmat = gl2cv(org_transform_matrix, return_cmat=True)
        # link distorted images
        source_image_path = Path(
            org_scene_root, modality, "resized_images", frame["file_path"]
        )
        if not source_image_path.exists():
            if frame["file_path"] in test_frames:
                logger.warning(f"Missing eval frame: {frame_name}")
                continue
            else:
                raise FileNotFoundError(f"Source path missing: {source_image_path}")
        target_image_path = f"{org2wai['resized_images']}/{frame_name}.jpg"
        os.symlink(source_image_path, out_path / target_image_path)
        wai_frame["image_distorted"] = target_image_path
        wai_frame["file_path"] = target_image_path

        # link anon_mask
        if has_mask and "mask_path" in frame:
            source_mask_path = Path(
                org_scene_root, modality, "resized_anon_masks", frame["mask_path"]
            )
            if not source_mask_path.exists():
                if frame["file_path"] in test_frames:
                    logger.info(f"Missing eval frame: {frame_name}")
                    continue
                else:
                    raise FileNotFoundError(f"Source path missing: {source_mask_path}")
            else:
                target_mask_path = f"{org2wai['resized_anon_masks']}/{frame_name}.png"
                os.symlink(source_mask_path, out_path / target_mask_path)
                wai_frame["anon_mask_distorted"] = target_mask_path

        wai_frame["transform_matrix"] = opencv_pose.tolist()

        # optional per-frame intrinsics
        for camera_key in CAMERA_KEYS:
            if camera_key in frame:
                wai_frame[camera_key] = frame[camera_key]

        # optional other keys
        other_keys = ["is_bad"]
        for other_key in other_keys:
            if other_key in frame:
                wai_frame[other_key] = frame[other_key]

        # Not used, revisit when generating split files!
        if frame["file_path"] in test_frames:
            eval_frame_names.append(frame_name)
        else:
            train_frame_names.append(frame_name)
        wai_frames.append(wai_frame)

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": True,
        "camera_model": meta["camera_model"],
        "camera_convention": "opencv",
        "scale_type": "metric",
        "frames": wai_frames,
        "frame_modalities": {
            "image_distorted": {"frame_key": "image_distorted", "format": "image"},
            "anon_mask_distorted": {
                "frame_key": "anon_mask_distorted",
                "format": "binary",
            },
        },
    }
    for camera_key in CAMERA_KEYS:
        if camera_key in meta:
            scene_meta[camera_key] = meta[camera_key]

    # init scene modalities
    scene_modalities = {}

    # link original COLMAP folder
    os.symlink(
        org_scene_root / modality / "colmap",
        out_path / "colmap",
        target_is_directory=True,
    )
    scene_modalities["colmap"] = {
        "cameras": {
            "path": "colmap/cameras.txt",
            "format": "readable",
        },
        "images": {
            "path": "colmap/images.txt",
            "format": "readable",
        },
        "points3D": {
            "path": "colmap/points3D.txt",
            "format": "readable",
        },
    }

    # process scene mesh and semantics (if available)
    source_scan_path = Path(org_scene_root) / "scans"
    if source_scan_path.exists():
        # load original mesh
        source_scan_mesh_path = Path(source_scan_path) / "mesh_aligned_0.05.ply"
        source_mesh_data = load_data(source_scan_mesh_path)
        # get data from mesh
        source_mesh_vertices = np.asarray(source_mesh_data.vertices)
        source_mesh_faces = np.asarray(source_mesh_data.faces)
        source_mesh_vertices_color = np.asarray(source_mesh_data.visual.vertex_colors)

        # map semantics if available
        source_segments_json_path = Path(source_scan_path) / "segments.json"
        if source_segments_json_path.exists():
            scene_annotations_exist = True
        else:
            scene_annotations_exist = False

        # process annotations and map onto the mesh
        if scene_annotations_exist:
            source_anno_json_path = Path(source_scan_path) / "segments_anno.json"
            source_segments = load_data(source_segments_json_path)
            source_anno = load_data(source_anno_json_path)

            # map segments to semantic classes (str->int)
            mapped_source_anno, scene_semantic_class_mapping = (
                map_semantic_class_to_index(
                    source_anno, scannet_semantic_class_mappings
                )
            )

            # get vertex semantic classes and instances
            vertices_semantic_class_id, vertices_instance_id = (
                map_semantics_on_vertices(source_segments, mapped_source_anno)
            )

            # get mesh semantic colors
            vertices_semantic_class_color = semantic_color_mapping[
                vertices_semantic_class_id
            ]
            vertices_instance_color = semantic_color_mapping[vertices_instance_id]

        # convert ScanNet++ mesh to OpenCV coordinate system
        vertices_homogeneous = np.hstack(
            (
                source_mesh_vertices,
                np.ones((source_mesh_vertices.shape[0], 1)),
            )
        )
        transform = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        transformed_vertices_homogeneous = vertices_homogeneous @ transform.T
        transformed_vertices = transformed_vertices_homogeneous[:, :3]

        # prepare data dict to be saved as labeled mesh (.ply)
        scene_modalities["labeled_mesh"] = {
            "scene_key": "labeled_mesh.ply",
            "format": "labeled_mesh",
        }
        labeled_mesh_data = {
            "vertices": transformed_vertices,
            "faces": source_mesh_faces,
            "vertices_color": source_mesh_vertices_color,
        }
        # add semantics annotations
        if scene_annotations_exist:
            labeled_mesh_data["vertices_semantic_class_id"] = vertices_semantic_class_id
            labeled_mesh_data["vertices_instance_id"] = vertices_instance_id
            labeled_mesh_data["vertices_semantic_class_color"] = (
                vertices_semantic_class_color.astype(np.uint8)
            )
            labeled_mesh_data["vertices_instance_color"] = (
                vertices_instance_color.astype(np.uint8)
            )
        # store
        store_data(out_path / "labeled_mesh.ply", labeled_mesh_data, "labeled_mesh")

        # add color to semantic class mapping
        if scene_annotations_exist:
            for semantic_class_id in scene_semantic_class_mapping.keys():
                semantic_class_color = semantic_color_mapping[int(semantic_class_id)]
                scene_semantic_class_mapping[semantic_class_id]["color"] = (
                    semantic_class_color.tolist()
                )

            # create instance mapping with id -> color
            scene_instance_mapping = {}
            for instance_id in np.unique(vertices_instance_id):
                instance_color = semantic_color_mapping[instance_id]
                scene_instance_mapping[str(instance_id)] = {
                    "color": instance_color.tolist()
                }

            # sort the mappings for readability
            scene_semantic_class_mapping = dict(
                sorted(
                    scene_semantic_class_mapping.items(), key=lambda item: int(item[0])
                )
            )
            scene_instance_mapping = dict(
                sorted(scene_instance_mapping.items(), key=lambda item: int(item[0]))
            )

            # Save semantic mappings
            mappings_out_path = Path(out_path) / "mappings"
            mappings_out_path.mkdir(parents=True, exist_ok=True)
            store_data(
                mappings_out_path / "semantic_class_mapping.json",
                scene_semantic_class_mapping,
                "readable",
            )
            store_data(
                mappings_out_path / "instance_mapping.json",
                scene_instance_mapping,
                "readable",
            )
            # Add semantic mappings to scene modalities
            scene_modalities["mappings"] = {
                "semantic_class": {
                    "path": "mappings/semantic_class_mapping.json",
                    "format": "readable",
                },
                "instance": {
                    "path": "mappings/instance_mapping.json",
                    "format": "readable",
                },
            }

    # update scene modalities in scene meta
    scene_meta["scene_modalities"] = scene_modalities

    ## TODO: This is only the pose of the final frame
    scene_meta["_applied_transform"] = gl2cv_cmat.tolist()
    scene_meta["_applied_transforms"] = {"opengl2opencv": gl2cv_cmat.tolist()}
    # save updated scene meta
    store_data(out_path / "scene_meta_distorted.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/scannetppv2.yaml")
    os.makedirs(cfg.root, exist_ok=True)
    scannet_semantic_class_mappings = get_semantic_class_mapping(cfg)

    # Preload semantic colors
    semantic_color_mapping = load_semantic_color_mapping()

    for modality in cfg.modalities:
        if modality != "dslr":
            raise NotImplementedError("Only DSLR support is implemented")
        convert_scenes_wrapper(
            convert_scene,
            cfg,
            # Use default get scenes
            get_original_scene_names_func=get_original_scene_names,
            # **kwargs passed to convert scene
            modality=modality,
            scannet_semantic_class_mappings=scannet_semantic_class_mappings,
            semantic_color_mapping=semantic_color_mapping,
        )
