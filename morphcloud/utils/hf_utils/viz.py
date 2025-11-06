# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utility functions for Gradio demo visualizations
"""

import copy
import os
from typing import Tuple

import cv2
import matplotlib
import numpy as np
import requests
import trimesh
from scipy.spatial.transform import Rotation


def remove_unreferenced_vertices(
    faces: np.ndarray, *vertice_attrs, return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Remove unreferenced vertices of a mesh.
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    Args:
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes

    Returns:
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (np.ndarray, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = np.unique(faces, return_inverse=True)
    faces = inv_map.astype(np.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def triangulate(
    faces: np.ndarray, vertices: np.ndarray = None, backslash: np.ndarray = None
) -> np.ndarray:
    """
    Triangulate a polygonal mesh.

    Args:
        faces (np.ndarray): [L, P] polygonal faces
        vertices (np.ndarray, optional): [N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        backslash (np.ndarray, optional): [L] boolean array indicating
            how to triangulate the quad faces. Defaults to None.

    Returns:
        (np.ndarray): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if vertices is not None:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if backslash is None:
            backslash = np.linalg.norm(
                vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=-1
            ) < np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], axis=-1)
    if backslash is None:
        loop_indice = np.stack(
            [
                np.zeros(P - 2, dtype=int),
                np.arange(1, P - 1, 1, dtype=int),
                np.arange(2, P, 1, dtype=int),
            ],
            axis=1,
        )
        return faces[:, loop_indice].reshape((-1, 3))
    else:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        faces = np.where(
            backslash[:, None],
            faces[:, [0, 1, 2, 0, 2, 3]],
            faces[:, [0, 1, 3, 3, 1, 2]],
        ).reshape((-1, 3))
        return faces


def image_mesh(
    *image_attrs: np.ndarray,
    mask: np.ndarray = None,
    tri: bool = False,
    return_indices: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    Args:
        *image_attrs (np.ndarray): image attributes in shape (height, width, [channels])
        mask (np.ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        faces (np.ndarray): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
        *vertex_attrs (np.ndarray): vertex attributes in corresponding order with input image_attrs
        indices (np.ndarray, optional): indices of vertices in the original mesh
    """
    assert (len(image_attrs) > 0) or (mask is not None), (
        "At least one of image_attrs or mask should be provided"
    )
    height, width = next(image_attrs).shape[:2] if mask is None else mask.shape
    assert all(img.shape[:2] == (height, width) for img in image_attrs), (
        "All image_attrs should have the same shape"
    )

    row_faces = np.stack(
        [
            np.arange(0, width - 1, dtype=np.int32),
            np.arange(width, 2 * width - 1, dtype=np.int32),
            np.arange(1 + width, 2 * width, dtype=np.int32),
            np.arange(1, width, dtype=np.int32),
        ],
        axis=1,
    )
    faces = (
        np.arange(0, (height - 1) * width, width, dtype=np.int32)[:, None, None]
        + row_faces[None, :, :]
    ).reshape((-1, 4))
    if mask is None:
        if tri:
            faces = triangulate(faces)
        ret = [faces, *(img.reshape(-1, *img.shape[2:]) for img in image_attrs)]
        if return_indices:
            ret.append(np.arange(height * width, dtype=np.int32))
        return tuple(ret)
    else:
        quad_mask = (
            mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
        ).ravel()
        faces = faces[quad_mask]
        if tri:
            faces = triangulate(faces)
        return remove_unreferenced_vertices(
            faces,
            *(x.reshape(-1, *x.shape[2:]) for x in image_attrs),
            return_indices=return_indices,
        )


def predictions_to_glb(
    predictions,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_ambiguous=False,
    as_mesh=True,
    conf_percentile=None,
) -> trimesh.Scene:
    """
    Converts MapAnything predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_ambiguous (bool): Apply final mask to filter ambiguous predictions (default: False)
        as_mesh (bool): Represent the data as a mesh instead of point cloud (default: False)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud/mesh and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    # Always use Pointmap Branch
    print("Using Pointmap Branch")
    if "world_points" not in predictions:
        raise ValueError(
            "world_points not found in predictions. Pointmap Branch requires 'world_points' key. "
            "Depthmap and Camera branches have been removed."
        )

    pred_world_points = predictions["world_points"]

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Create mask for filtering
    mask = np.ones(len(vertices_3d), dtype=bool)
    final_mask = predictions["final_mask"].reshape(-1)

    # Confidence masking
    if conf_percentile is not None and "conf" in predictions:
        # print ("Applying confidence masking...")
        conf = predictions["conf"].reshape(-1)
        threshold = np.percentile(conf, conf_percentile)
        # print (f"Confidence threshold at {conf_percentile} percentile: {threshold}")
        conf_mask = conf >= threshold
        mask = mask & conf_mask

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        mask = mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = (
            (colors_rgb[:, 0] > 240)
            & (colors_rgb[:, 1] > 240)
            & (colors_rgb[:, 2] > 240)
        )
        mask = mask & ~white_bg_mask

    # Use final_mask when mask_ambiguous is checked
    if mask_ambiguous:
        mask = mask & final_mask

    vertices_3d = vertices_3d[mask].copy()
    colors_rgb = colors_rgb[mask].copy()

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    if as_mesh:
        # Create mesh from pointcloud
        # try:
        if selected_frame_idx is not None:
            # Single frame case - we can create a proper mesh
            H, W = pred_world_points.shape[1:3]

            # Get original unfiltered data for mesh creation
            original_points = pred_world_points.reshape(H, W, 3)

            # Reshape original image data properly
            if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
                original_image_colors = np.transpose(images[0], (1, 2, 0))
            else:  # Assume already in HWC format
                original_image_colors = images[0]
            original_image_colors *= 255
            # Get original final mask
            original_final_mask = predictions["final_mask"][selected_frame_idx].reshape(
                H, W
            )

            # Create mask based on final mask
            mask = original_final_mask

            # Confidence masking
            if conf_percentile is not None and "conf" in predictions:
                # print ("Applying confidence masking...")
                conf = predictions["conf"][selected_frame_idx].reshape(-1)
                threshold = np.percentile(conf, conf_percentile)
                # print (f"Confidence threshold at {conf_percentile} percentile: {threshold}")
                conf_mask = conf >= threshold
                mask = mask & conf_mask.reshape(H, W)

            # Additional background masks if needed
            if mask_black_bg:
                black_bg_mask = original_image_colors.sum(axis=2) >= 16
                mask = mask & black_bg_mask

            if mask_white_bg:
                white_bg_mask = ~(
                    (original_image_colors[:, :, 0] > 240)
                    & (original_image_colors[:, :, 1] > 240)
                    & (original_image_colors[:, :, 2] > 240)
                )
                mask = mask & white_bg_mask

            # Check if normals are available in predictions
            vertex_normals = None
            if "normal" in predictions and predictions["normal"] is not None:
                # Get normals for the selected frame
                frame_normals = (
                    predictions["normal"][selected_frame_idx]
                    if selected_frame_idx is not None
                    else predictions["normal"][0]
                )

                # Create faces and vertices using image_mesh with normals support
                faces, vertices, vertex_colors, vertex_normals = image_mesh(
                    original_points * np.array([1, -1, 1], dtype=np.float32),
                    original_image_colors / 255.0,
                    frame_normals * np.array([1, -1, 1], dtype=np.float32),
                    mask=mask,
                    tri=True,
                    return_indices=False,
                )

                # Apply coordinate transformations to normals
                vertex_normals = vertex_normals * np.array([1, -1, 1], dtype=np.float32)
            else:
                # Create faces and vertices using image_mesh without normals
                faces, vertices, vertex_colors = image_mesh(
                    original_points * np.array([1, -1, 1], dtype=np.float32),
                    original_image_colors / 255.0,
                    mask=mask,
                    tri=True,
                    return_indices=False,
                )

            # vertices = vertices * np.array([1, -1, 1], dtype=np.float32)

            # Create trimesh object with optional normals
            mesh_data = trimesh.Trimesh(
                vertices=vertices * np.array([1, -1, 1], dtype=np.float32),
                faces=faces,
                vertex_colors=(vertex_colors * 255).astype(np.uint8),
                vertex_normals=(vertex_normals if vertex_normals is not None else None),
                process=False,
            )
            scene_3d.add_geometry(mesh_data)

        else:
            # Multi-frame case - create separate meshes for each frame
            print("Creating mesh for multi-frame data...")

            for frame_idx in range(pred_world_points.shape[0]):
                H, W = pred_world_points.shape[1:3]

                # Get data for this frame
                frame_points = pred_world_points[frame_idx]
                frame_final_mask = predictions["final_mask"][frame_idx]

                # Get frame image
                if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
                    frame_image = np.transpose(images[frame_idx], (1, 2, 0))
                else:  # Assume already in HWC format
                    frame_image = images[frame_idx]
                frame_image *= 255
                # Create mask for this frame using final_mask
                mask = frame_final_mask

                # Additional background masks if needed
                if mask_black_bg:
                    black_bg_mask = frame_image.sum(axis=2) >= 16
                    mask = mask & black_bg_mask

                if mask_white_bg:
                    white_bg_mask = ~(
                        (frame_image[:, :, 0] > 240)
                        & (frame_image[:, :, 1] > 240)
                        & (frame_image[:, :, 2] > 240)
                    )
                    mask = mask & white_bg_mask
                if conf_percentile is not None and "conf" in predictions:
                    # print ("Applying confidence masking...")
                    conf = predictions["conf"][frame_idx].reshape(-1)
                    threshold = np.percentile(conf, conf_percentile)
                    # print (f"Confidence threshold at {conf_percentile} percentile: {threshold}")
                    conf_mask = conf >= threshold
                    mask = mask & conf_mask.reshape(H, W)
                # Create mesh for this frame
                faces, vertices, vertex_colors = image_mesh(
                    frame_points * np.array([1, -1, 1], dtype=np.float32),
                    frame_image / 255.0,
                    mask=mask,
                    tri=True,
                    return_indices=False,
                )

                vertices = vertices * np.array([1, -1, 1], dtype=np.float32)
                # Create trimesh object for this frame
                frame_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=(vertex_colors * 255).astype(np.uint8),
                    process=False,
                )
                scene_3d.add_geometry(frame_mesh)
    else:
        point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
        scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = camera_matrices[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(
                scene_3d, world_to_camera, current_color, scene_scale
            )

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, camera_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """
    scene_scale = 12
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1
    # cam_width = scene_scale * 0.05
    # cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(
    scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray
) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 0, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = (
        np.linalg.inv(extrinsics_matrices[0])
        @ opengl_conversion_matrix
        @ align_rotation
    )
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(
    transformation: np.ndarray, points: np.ndarray, dim: int = None
) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(
        -1, -2
    )  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
