# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import shutil
import traceback
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from argconf import argconf_parse
from torch.nn import functional as F
from tqdm import tqdm

from data_processing.wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from data_processing.wai_processing.utils.state import (
    SceneProcessLock,
    set_processing_state,
)
from mapanything.utils.wai.camera import ALL_CAM_PARAMS, cv2gl
from mapanything.utils.wai.core import (
    get_frame,
    load_data,
    load_frame,
    set_frame,
    store_data,
)
from mapanything.utils.wai.ops import to_dtype_device, to_torch_device_contiguous
from mapanything.utils.wai.scene_frame import get_scene_frame_names

logger = logging.getLogger(__name__)


def prepare_scene_rendering(mesh_data, camera_params):
    """Creates a pyrender scene to be rendered with pyrender"""

    import os

    # needed for pyrender scene rendering
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import pyrender

    # init dict to store rendering data
    rendering_data = {}

    # initialize renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=camera_params["w"], viewport_height=camera_params["h"]
    )
    rendering_data["renderer"] = renderer

    # load or create a pyrender scene
    if isinstance(mesh_data, trimesh.Trimesh):
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh_data)
        scene.add(mesh)
    elif isinstance(mesh_data, trimesh.Scene):
        scene = pyrender.Scene.from_trimesh_scene(mesh_data)
    else:
        raise ValueError(
            "Scene rendering currently supports only trimesh.Trimesh or trimesh.Scene meshes."
        )

    # add camera
    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["fl_x"],
        fy=camera_params["fl_y"],
        cx=camera_params["cx"],
        cy=camera_params["cy"],
    )
    camera_node = scene.add(camera)
    rendering_data["camera_node"] = camera_node

    # add scene
    rendering_data["scene"] = scene

    return rendering_data


def render_scene(rendering_data, c2w_gl):
    """Renders a pyrender scene using pyrender"""
    from pyrender.constants import RenderFlags

    rendering_data["scene"].set_pose(
        rendering_data["camera_node"],
        pose=to_dtype_device(c2w_gl, device=np.ndarray, dtype=np.float32),
    )

    color, depth = rendering_data["renderer"].render(
        rendering_data["scene"], flags=RenderFlags.FLAT
    )
    color = color / 255.0  # rgb in [0,1]
    return color, depth


def prepare_mesh_rendering(mesh_data, camera_data, device="cuda"):
    """Prepares the rendering data for rendering with nvdiffrast"""

    # check if the mesh data is a wai labeled mesh
    if (
        isinstance(mesh_data, dict)
        and "is_labeled_mesh" in mesh_data
        and mesh_data["is_labeled_mesh"]
    ):
        # mesh is a wai labeled mesh, copy its contents
        rendering_data = mesh_data.copy()
        # normalize color to [0,1]
        rendering_data["vertices_color"] = rendering_data["vertices_color"] / 255.0

    elif isinstance(mesh_data, trimesh.Trimesh):
        # init dict to store rendering data
        rendering_data = {}

        # get mesh vertices and faces
        vertices = np.asarray(mesh_data.vertices, dtype=np.float32)
        faces = np.asarray(mesh_data.faces, dtype=np.int32)
        rendering_data["vertices"] = vertices
        rendering_data["faces"] = faces

        # get vertices color if available
        if (
            hasattr(mesh_data, "visual")
            and hasattr(mesh_data.visual, "vertex_colors")
            and mesh_data.visual.vertex_colors is not None
        ):
            vertices_color = np.asarray(
                mesh_data.visual.vertex_colors, dtype=np.float32
            )[:, :3]  # discard alpha channel
            rendering_data["vertices_color"] = vertices_color / 255.0

        # get texture if available
        if (
            hasattr(mesh_data, "visual")
            and hasattr(mesh_data.visual, "material")
            and mesh_data.visual.material.image is not None
        ):
            texture_image = np.array(mesh_data.visual.material.image, dtype=np.float32)
            vertices_uvs = np.asarray(mesh_data.visual.uv, dtype=np.float32)

            # normalize color in [0,1]
            texture_image = texture_image / 255.0

            # add texture data to rendering data
            rendering_data["texture"] = {
                "image": texture_image,
                "vertices_uvs": vertices_uvs,
            }

    # sanity check
    if not any(k in rendering_data for k in ["texture", "vertices_color"]):
        raise ValueError(
            "Rendering requires mesh data to have texture and/or vertices color."
        )

    # convert data to torch and load on device
    rendering_data = to_torch_device_contiguous(rendering_data, device, contiguous=True)

    # add nvdiffrast rasterizer (OpenGL)
    rasterizer = dr.RasterizeGLContext()
    rendering_data["rasterizer"] = rasterizer

    # add image size and camera intrinsics
    rendering_data["w"] = camera_data["w"]
    rendering_data["h"] = camera_data["h"]
    rendering_data["fl_x"] = camera_data["fl_x"]
    rendering_data["fl_y"] = camera_data["fl_y"]
    rendering_data["cx"] = camera_data["cx"]
    rendering_data["cy"] = camera_data["cy"]

    return rendering_data


def render_mesh(rendering_data, c2w_gl, invalid_face_id, near, far, device="cuda"):
    """Renders a mesh using nvdiffrast, producing color, depth and face_ids"""
    import nvdiffrast.torch as dr

    # initialize rasterization
    x = rendering_data["w"] / (2.0 * rendering_data["fl_x"])
    y = rendering_data["h"] / (2.0 * rendering_data["fl_y"])
    projection = np.array(
        [
            [1 / x, 0, 0, 0],
            [0, -1 / y, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    projection = torch.from_numpy(projection).to(device)
    c2w_gl = torch.from_numpy(c2w_gl).to(device)
    view_matrix = projection @ torch.inverse(c2w_gl)

    # rasterize mesh (get face id for each pixel)
    vertices_h = F.pad(
        rendering_data["vertices"], pad=(0, 1), mode="constant", value=1.0
    )
    vertices_clip = torch.matmul(
        vertices_h, torch.transpose(view_matrix, 0, 1)
    ).unsqueeze(0)  # [1, num_vertices, 4]
    rasterization, _ = dr.rasterize(
        rendering_data["rasterizer"],
        vertices_clip,
        rendering_data["faces"],
        (rendering_data["h"], rendering_data["w"]),
    )  # [1, h, w, 4]
    unbatched_rasterization = rasterization.squeeze(0)  # [h, w, 4]

    # render color (priority to texture over vertices color)
    if "texture" in rendering_data:
        vertices_uvs = rendering_data["texture"]["vertices_uvs"]

        # invert v coordinate (for nvdiffrast)
        vertices_uvs[:, 1] = 1.0 - vertices_uvs[:, 1]

        # interpolate UVs
        uv_interp, _ = dr.interpolate(
            vertices_uvs,
            rasterization,
            rendering_data["faces"],
        )

        # render texture
        color = dr.texture(
            rendering_data["texture"]["image"].unsqueeze(0),
            uv_interp,
            filter_mode="linear",
        )

    elif "vertices_color" in rendering_data:
        # interpolate vertices color
        color, _ = dr.interpolate(
            rendering_data["vertices_color"].float(),
            rasterization,
            rendering_data["faces"],
        )
    else:
        raise ValueError("Rendering requires texture and/or vertices color.")

    # postprocess faces ids (rasterized faces ids have an offset of +1, valid faces ids are >= 1)
    rasterized_face_id = unbatched_rasterization[..., 3].int()
    valid_faces = rasterized_face_id >= 1

    # initialize output faces ids as INVALID_FACE_ID
    output_face_id = torch.full_like(rasterized_face_id, fill_value=invalid_face_id)

    # fill valid faces ids in the output, removing the offset
    output_face_id[valid_faces] = rasterized_face_id[valid_faces] - 1

    # get depth (rasterized depth is in clip space z/w, [-1, 1] range)
    clip_depth = unbatched_rasterization[..., 2]

    # convert clip_depth to metric depth (initialize as invalid depth = 0)
    depth = torch.zeros_like(clip_depth)

    # avoid numerical issues around far plane
    valid_depth = clip_depth < 0.999

    # compute metric depth
    valid_pixels = valid_faces & valid_depth
    depth[valid_pixels] = (2.0 * near * far) / (
        far + near - clip_depth[valid_pixels] * (far - near)
    )

    # ouput data as numpy arrays
    color = color.squeeze(0).cpu().numpy()
    depth = depth.squeeze(0).cpu().numpy()
    output_face_id = output_face_id.squeeze(0).cpu().numpy()

    return color, depth, output_face_id


def rendering(cfg, scene_name, overwrite=False):
    scene_root = Path(cfg.root, scene_name)
    scene_meta = load_data(Path(scene_root, "scene_meta.json"), "scene_meta")
    camera_data = {k: v for k, v in scene_meta.items() if k in ALL_CAM_PARAMS}
    mesh_to_render_name, mesh_to_render_ext = cfg.mesh_to_render.split(".")

    # sanity checks
    if scene_meta["camera_model"] != "PINHOLE":
        raise ValueError(
            "Mesh rendering only supports pinhole camera models, run undistortion first!"
        )
    if not scene_meta["shared_intrinsics"]:
        raise NotImplementedError("We only support a single camera model per scene atm")
    if not torch.cuda.is_available() and cfg.rendering_type == "labeled_mesh":
        raise ValueError("Labeled mesh rendering is only supported on GPU.")
    if mesh_to_render_name not in scene_meta["scene_modalities"]:
        raise ValueError(
            f"Mesh {mesh_to_render_name} is not available in scene_modalities."
        )
    if (
        cfg.rendering_type == "scene"
        and "rendered_face_ids" in cfg.frame_modalities_to_render
    ):
        raise ValueError("Rendering face ids is not supported in scene rendering.")
    if any(
        fmr not in cfg.supported_frame_modalities_to_render
        for fmr in cfg.frame_modalities_to_render
    ):
        raise ValueError(
            f"Found unsupported frame modalities to render. Allowed modalities are: {cfg.supported_frame_modalities_to_render}"
        )
    if mesh_to_render_ext not in cfg.supported_mesh_formats:
        raise ValueError(
            f"Unsupported mesh format .{mesh_to_render_ext}. Supported formats are: {cfg.supported_mesh_formats}"
        )

    # create output directories
    for fmr in cfg.frame_modalities_to_render:
        fmr_dir_path = Path(scene_root, f"rendered_{fmr}")
        if Path(fmr_dir_path).exists():
            if overwrite:
                shutil.rmtree(fmr_dir_path)
            else:
                raise FileExistsError(f"Path already exists: {fmr_dir_path} ")

    # load mesh data
    mesh_path = Path(
        scene_root, scene_meta["scene_modalities"][mesh_to_render_name]["scene_key"]
    )
    if cfg.rendering_type == "labeled_mesh":
        mesh_data = load_data(mesh_path, format="labeled_mesh")
    elif cfg.rendering_type in ["mesh", "scene"]:
        # generic mesh loader
        mesh_data = load_data(mesh_path, format="mesh")
    else:
        raise ValueError(f"Unsupported rendering type: {cfg.rendering_type}")

    # prepare rendering
    if cfg.rendering_type in ["mesh", "labeled_mesh"]:
        scene_rendering_data = prepare_mesh_rendering(mesh_data, camera_data)
    elif cfg.rendering_type == "scene":
        scene_rendering_data = prepare_scene_rendering(mesh_data, camera_data)

    # render scene frames
    for frame_name in tqdm(scene_frame_names[scene_name]):
        sample = load_frame(scene_root, frame_name)
        c2w = sample["extrinsics"].numpy()
        if cfg.get("mesh_transform") is not None:
            # apply the inverse of the mesh_transform to extrinsics
            c2w = np.linalg.inv(cfg.mesh_transform) @ c2w
        # gl convention needed for rendering (nvdiffrast and pyrender)
        c2w_gl = cv2gl(c2w)

        # render
        if cfg.rendering_type in ["mesh", "labeled_mesh"]:
            color, depth, face_id = render_mesh(
                scene_rendering_data,
                c2w_gl,
                cfg.invalid_face_id,
                cfg.near,
                cfg.far,
            )
        elif cfg.rendering_type == "scene":
            color, depth = render_scene(scene_rendering_data, c2w_gl)

        # --- update frame scene_meta ---
        frame = get_frame(scene_meta, frame_name)

        # rendered depth
        if "rendered_depth" in cfg.frame_modalities_to_render:
            rel_depth_frame_path = f"{'rendered_depth'}/{frame_name}.exr"
            store_data(
                scene_root / rel_depth_frame_path,
                depth,
                "depth",
            )
            frame["rendered_depth"] = rel_depth_frame_path

        # rendered image
        if "rendered_image" in cfg.frame_modalities_to_render:
            rel_color_frame_path = f"{'rendered_image'}/{frame_name}.png"
            store_data(
                scene_root / rel_color_frame_path,
                color,
                "image",
            )
            frame["rendered_image"] = rel_color_frame_path

        # rendered mesh faces
        if "rendered_mesh_faces" in cfg.frame_modalities_to_render:
            rel_mesh_faces_frame_path = f"{'rendered_mesh_faces'}/{frame_name}.npz"
            store_data(
                scene_root / rel_mesh_faces_frame_path,
                face_id,
                "numpy",
            )
            frame["rendered_mesh_faces"] = rel_mesh_faces_frame_path

        # update frame data in scene_meta
        set_frame(scene_meta, frame_name, frame, sort=True)

    # --- update frame_modalities ---
    frame_modalities = scene_meta["frame_modalities"]
    if "rendered_depth" in cfg.frame_modalities_to_render:
        frame_modalities["rendered_depth"] = {
            "frame_key": "rendered_depth",
            "format": "depth",
        }
    if "rendered_image" in cfg.frame_modalities_to_render:
        frame_modalities["rendered_image"] = {
            "frame_key": "rendered_image",
            "format": "image",
        }
    if "rendered_mesh_faces" in cfg.frame_modalities_to_render:
        frame_modalities["rendered_mesh_faces"] = {
            "frame_key": "rendered_mesh_faces",
            "format": "numpy",
        }

    scene_meta["frame_modalities"] = frame_modalities

    # store new scene_meta
    store_data(Path(cfg.root, scene_name, "scene_meta.json"), scene_meta, "scene_meta")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = argconf_parse(str(Path(WAI_PROC_CONFIG_PATH, "rendering.yaml")))
    if cfg.get("root") is None:
        logger.info(
            "Specify the root via: 'python scripts/run_rendering.py root=<root_path>'"
        )

    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled")

    scene_frame_names = get_scene_frame_names(cfg)

    for scene_name in tqdm(scene_frame_names):
        try:
            scene_root = Path(cfg.root, scene_name)
            with SceneProcessLock(scene_root):
                logger.info(f"Processing: {scene_name}")
                set_processing_state(scene_root, "rendering", "running")
                rendering(cfg, scene_name, overwrite=overwrite)
                set_processing_state(scene_root, "rendering", "finished")
        except Exception:
            logger.error(f"Rendering failed on scene: {scene_name}")
            trace_message = traceback.format_exc()
            logger.error(trace_message)
            set_processing_state(
                scene_root, "rendering", "failed", message=trace_message
            )
            continue
