# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
from einops import rearrange, repeat
from tqdm import tqdm


def create_frustum_from_intrinsics(
    intrinsics: torch.Tensor,
    near: torch.Tensor | float,
    far: torch.Tensor | float,
) -> torch.Tensor:
    r"""
    Create a frustum from camera intrinsics.

    Args:
        intrinsics (torch.Tensor): Bx3x3 Intrinsics of cameras.
        near (torch.Tensor or float): [B] Near plane distance.
        far (torch.Tensor or float): [B] Far plane distance.

    Returns:
        frustum (torch.Tensor): Bx8x3 batch of frustum points following the order:
            5 ---------- 4
            |\          /|
            6 \        / 7
             \ 1 ---- 0 /
              \|      |/
               2 ---- 3
    """

    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    # Calculate the offsets at the near plane
    near_x = near * (cx / fx)
    near_y = near * (cy / fy)
    far_x = far * (cx / fx)
    far_y = far * (cy / fy)

    # Define frustum vertices in camera space
    near_plane = torch.stack(
        [
            torch.stack([near_x, near_y, near * torch.ones_like(near_x)], dim=-1),
            torch.stack([-near_x, near_y, near * torch.ones_like(near_x)], dim=-1),
            torch.stack([-near_x, -near_y, near * torch.ones_like(near_x)], dim=-1),
            torch.stack([near_x, -near_y, near * torch.ones_like(near_x)], dim=-1),
        ],
        dim=1,
    )

    far_plane = torch.stack(
        [
            torch.stack([far_x, far_y, far * torch.ones_like(far_x)], dim=-1),
            torch.stack([-far_x, far_y, far * torch.ones_like(far_x)], dim=-1),
            torch.stack([-far_x, -far_y, far * torch.ones_like(far_x)], dim=-1),
            torch.stack([far_x, -far_y, far * torch.ones_like(far_x)], dim=-1),
        ],
        dim=1,
    )

    return torch.cat([near_plane, far_plane], dim=1)


def _frustum_to_triangles(frustum: torch.Tensor) -> torch.Tensor:
    """
    Convert frustum to triangles.

    Args:
        frustums (torch.Tensor): Bx8 batch of frustum points.

    Returns:
        frustum_triangles (torch.Tensor): Bx3x3 batch of frustum triangles.
    """

    triangle_inds = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
            [1, 4, 5],
            [1, 0, 4],
            [2, 6, 7],
            [2, 3, 7],
            [6, 7, 4],
            [6, 5, 4],
        ]
    )
    frustum_triangles = frustum[:, triangle_inds]
    return frustum_triangles


def segment_triangle_intersection_check(
    start_points: torch.Tensor,
    end_points: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """
    Check if segments (lines with starting and end point) intersect triangles in 3D using the
    Moller-Trumbore algorithm.

    Args:
        start_points (torch.Tensor): Bx3 Starting points of the segment.
        end_points (torch.Tensor): Bx3 End points of the segment.
        triangles (torch.Tensor): Bx3x3 Vertices of the triangles.

    Returns:
        intersects (torch.Tensor): B Boolean tensor indicating if each ray intersects its
        corresponding triangle.
    """
    vertex0 = triangles[:, 0, :]
    vertex1 = triangles[:, 1, :]
    vertex2 = triangles[:, 2, :]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    ray_vectors = end_points - start_points
    max_lengths = torch.norm(ray_vectors, dim=1)
    ray_vectors = ray_vectors / max_lengths[:, None]
    h = torch.cross(ray_vectors, edge2, dim=1)
    a = (edge1 * h).sum(dim=1)

    epsilon = 1e-6
    mask = torch.abs(a) > epsilon
    f = torch.zeros_like(a)
    f[mask] = 1.0 / a[mask]

    s = start_points - vertex0
    u = f * (s * h).sum(dim=1)
    q = torch.cross(s, edge1, dim=1)
    v = f * (ray_vectors * q).sum(dim=1)

    t = f * (edge2 * q).sum(dim=1)

    # Check conditions
    intersects = (
        (u >= 0)
        & (u <= 1)
        & (v >= 0)
        & (u + v <= 1)
        & (t >= epsilon)
        & (t <= max_lengths)
    )

    return intersects


def triangle_intersection_check(
    triangles1: torch.Tensor,
    triangles2: torch.Tensor,
) -> torch.Tensor:
    """
    Check if two triangles intersect.

    Args:
        triangles1 (torch.Tensor): Bx3x3 Vertices of the first batch of triangles.
        triangles2 (torch.Tensor): Bx3x3 Vertices of the first batch of triangles.

    Returns:
        triangle_intersection (torch.Tensor): B Boolean tensor indicating if triangles intersect.
    """
    n = triangles1.shape[1]
    start_points1 = rearrange(triangles1, "B N C -> (B N) C")
    end_points1 = rearrange(
        triangles1[:, torch.arange(1, n + 1) % n], "B N C -> (B N) C"
    )

    start_points2 = rearrange(triangles2, "B N C -> (B N) C")
    end_points2 = rearrange(
        triangles2[:, torch.arange(1, n + 1) % n], "B N C -> (B N) C"
    )
    intersection_1_2 = segment_triangle_intersection_check(
        start_points1, end_points1, repeat(triangles2, "B N C -> (B N2) N C", N2=3)
    )
    intersection_2_1 = segment_triangle_intersection_check(
        start_points2, end_points2, repeat(triangles1, "B N C -> (B N2) N C", N2=3)
    )
    triangle_intersection = torch.any(
        rearrange(intersection_1_2, "(B N N2) -> B (N N2)", B=triangles1.shape[0], N=n),
        dim=1,
    ) | torch.any(
        rearrange(intersection_2_1, "(B N N2) -> B (N N2)", B=triangles1.shape[0], N=n),
        dim=1,
    )
    return triangle_intersection


def frustum_intersection_check(
    frustums: torch.Tensor,
    check_inside: bool = True,
    chunk_size: int = 500,
    device: str | None = None,
) -> torch.Tensor:
    """
    Check if any pair of the frustums intersect with each other.

    Args:
        frustums (torch.Tensor): Bx8 batch of frustum points.
        check_inside (bool): If True, also checks if one frustum is inside another.
            Defaults to True.
        chunk_size (Optional[int]): Number of chunks to split the computation into.
            Defaults to 500.
        device (Optional[str]): Device to store exhaustive frustum intersection matrix on.
            Defaults to None.

    Returns:
        frustum_intersection (torch.Tensor): BxB tensor of Booleans indicating if any pair
        of frustums intersect with each other.
    """
    B = frustums.shape[0]
    if device is None:
        device = frustums.device
    frustum_triangles = _frustum_to_triangles(frustums)
    T = frustum_triangles.shape[1]

    # Perform frustum in frustum check if required
    if check_inside:
        frustum_intersection = frustums_in_frustum_check(
            frustums=frustums, chunk_size=chunk_size, device=device
        )
    else:
        frustum_intersection = torch.zeros((B, B), dtype=torch.bool, device=device)

    # Check triangle intersections in chunks
    for i in tqdm(range(0, B, chunk_size), desc="Checking triangle intersections"):
        i_end = min(i + chunk_size, B)
        chunk_i_size = i_end - i

        for j in range(0, B, chunk_size):
            j_end = min(j + chunk_size, B)
            chunk_j_size = j_end - j

            # Process all triangle pairs between the two chunks in a vectorized way
            triangles_i = frustum_triangles[i:i_end]  # [chunk_i, T, 3, 3]
            triangles_j = frustum_triangles[j:j_end]  # [chunk_j, T, 3, 3]

            # Reshape to process all triangle pairs at once
            tri_i = triangles_i.reshape(chunk_i_size * T, 3, 3)
            tri_j = triangles_j.reshape(chunk_j_size * T, 3, 3)

            # Expand for all pairs - explicitly specify dimensions instead of using ...
            tri_i_exp = repeat(tri_i, "bt i j -> (bt bj_t) i j", bj_t=chunk_j_size * T)
            tri_j_exp = repeat(tri_j, "bt i j -> (bi_t bt) i j", bi_t=chunk_i_size * T)

            # Check intersection
            batch_intersect = triangle_intersection_check(tri_i_exp, tri_j_exp)

            # Reshape and check if any triangle pair intersects
            batch_intersect = batch_intersect.reshape(chunk_i_size, T, chunk_j_size, T)
            batch_intersect = batch_intersect.any(dim=(1, 3))

            # Update result
            frustum_intersection[i:i_end, j:j_end] |= batch_intersect.to(device)

    return frustum_intersection


def ray_triangle_intersection_check(
    ray_origins: torch.Tensor,
    ray_vectors: torch.Tensor,
    triangles: torch.Tensor,
    max_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Check if rays intersect triangles in 3D using the Moller-Trumbore algorithm, considering the
    finite length of rays.

    Args:
        ray_origins (torch.Tensor): Bx3 Origins of the rays.
        ray_vectors (torch.Tensor): Bx3 Direction vectors of the rays.
        triangles (torch.Tensor): Bx3x3 Vertices of the triangles.
        max_lengths Optional[torch.Tensor]: B Maximum lengths of the rays.

    Returns:
        intersects (torch.Tensor): B Boolean tensor indicating if each ray intersects its
        corresponding triangle.
    """
    vertex0 = triangles[:, 0, :]
    vertex1 = triangles[:, 1, :]
    vertex2 = triangles[:, 2, :]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = torch.cross(ray_vectors, edge2, dim=1)
    a = (edge1 * h).sum(dim=1)

    epsilon = 1e-6
    mask = torch.abs(a) > epsilon
    f = torch.zeros_like(a)
    f[mask] = 1.0 / a[mask]

    s = ray_origins - vertex0
    u = f * (s * h).sum(dim=1)
    q = torch.cross(s, edge1, dim=1)
    v = f * (ray_vectors * q).sum(dim=1)

    t = f * (edge2 * q).sum(dim=1)

    # Check conditions
    intersects = (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t >= epsilon)
    if max_lengths is not None:
        intersects &= t <= max_lengths

    return intersects


#### Checks for frustums
def _frustum_to_planes(frustums: torch.Tensor) -> torch.Tensor:
    r"""
    Converts frustum parameters to plane representation.

    Args:
        frustums (torch.Tensor): Bx8 batch of frustum points following the order:
            5 ---------- 4
            |\          /|
            6 \        / 7
             \ 1 ---- 0 /
              \|      |/
               2 ---- 3

    Returns:
        planes (torch.Tensor): Bx6x4 where 6 represents the six frustum planes and
                                    4 represents plane parameters [a, b, c, d].
    """
    planes = []
    for inds in [[0, 1, 3], [1, 6, 2], [0, 3, 7], [2, 6, 3], [0, 5, 1], [6, 5, 4]]:
        normal = torch.cross(
            frustums[:, inds[1]] - frustums[:, inds[0]],
            frustums[:, inds[2]] - frustums[:, inds[0]],
            dim=1,
        )
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        d = -torch.sum(normal * frustums[:, inds[0]], dim=1, keepdim=True)
        planes.append(torch.cat([normal, d], -1))
    return torch.stack(planes, 1)


def points_in_frustum_check(
    frustums: torch.Tensor,
    points: torch.Tensor,
    chunk_size: int | None = None,
    device: str | None = None,
):
    """
    Check if points are inside frustums.

    Args:
        frustums (torch.Tensor): Bx8 batch of frustum points.
        points (torch.Tensor): BxNx3 batch of points.
        chunk_size (Optional[int]): Number of chunks to split the computation into. Defaults to None.
        device (Optional[str]): Device to perform computation on. Defaults to None.

    Returns:
        inside (torch.Tensor): BxN batch of Booleans indicating if points are inside frustums.
    """
    if device is None:
        device = frustums.device

    if chunk_size is not None:
        # Split computation into chunks to avoid OOM errors for large batch sizes
        point_plane_direction = []
        for chunk_idx in range(0, frustums.shape[0], chunk_size):
            chunk_frustum_planes = _frustum_to_planes(
                frustums[chunk_idx : chunk_idx + chunk_size]
            )
            # Bx8x4 tensor of plane parameters [a, b, c, d]
            chunk_points = points[chunk_idx : chunk_idx + chunk_size]
            chunk_point_plane_direction = torch.einsum(
                "bij,bnj->bni", (chunk_frustum_planes[:, :, :-1], chunk_points)
            ) + repeat(
                chunk_frustum_planes[:, :, -1], "B P -> B N P", N=chunk_points.shape[1]
            )  # BxMxN tensor
            point_plane_direction.append(chunk_point_plane_direction.to(device))
        point_plane_direction = torch.cat(point_plane_direction)
    else:
        # Convert frustums to planes
        frustum_planes = _frustum_to_planes(
            frustums
        )  # Bx8x4 tensor of plane parameters [a, b, c, d]
        # Compute dot product between each point and each plane
        point_plane_direction = torch.einsum(
            "bij,bnj->bni", (frustum_planes[:, :, :-1], points)
        ) + repeat(frustum_planes[:, :, -1], "B P -> B N P", N=points.shape[1]).to(
            device
        )  # BxMxN tensor

    inside = (point_plane_direction >= 0).all(-1)
    return inside


def frustums_in_frustum_check(
    frustums: torch.Tensor,
    chunk_size: int,
    device: str | None = None,
    use_double_chunking: bool = True,
):
    """
    Check if frustums are contained within other frustums.

    Args:
        frustums (torch.Tensor): Bx8 batch of frustum points.
        chunk_size (Optional[int]): Number of chunks to split the computation into.
            Defaults to None.
        device (Optional[str]): Device to store exhaustive frustum containment matrix on.
            Defaults to None.
        use_double_chunking (bool): If True, use double chunking to avoid OOM errors.
            Defaults to True.

    Returns:
        frustum_contained (torch.Tensor): BxB batch of Booleans indicating if frustums are inside
        other frustums.
    """
    B = frustums.shape[0]
    if device is None:
        device = frustums.device

    if use_double_chunking:
        frustum_contained = torch.zeros((B, B), dtype=torch.bool, device=device)
        # Check if frustums are containing each other by processing in chunks
        for i in tqdm(range(0, B, chunk_size), desc="Checking frustum containment"):
            i_end = min(i + chunk_size, B)
            chunk_i_size = i_end - i

            for j in range(0, B, chunk_size):
                j_end = min(j + chunk_size, B)
                chunk_j_size = j_end - j

                # Process a chunk of frustums against another chunk
                frustums_i = frustums[i:i_end]
                frustums_j_vertices = frustums[
                    j:j_end, :1
                ]  # Just need one vertex to check containment

                # Perform points in frustum check
                contained = rearrange(
                    points_in_frustum_check(
                        repeat(frustums_i, "B ... -> (B B2) ...", B2=chunk_j_size),
                        repeat(
                            frustums_j_vertices, "B ... -> (B2 B) ...", B2=chunk_i_size
                        ),
                    )[:, 0],
                    "(B B2) -> B B2",
                    B=chunk_i_size,
                ).to(device)

                # Map results back to the full matrix
                frustum_contained[i:i_end, j:j_end] |= contained
                frustum_contained[j:j_end, i:i_end] |= contained.transpose(
                    0, 1
                )  # Symmetric relation
    else:
        # Perform points in frustum check with a single chunked loop
        frustum_contained = rearrange(
            points_in_frustum_check(
                repeat(frustums, "B ... -> (B B2) ...", B2=B),
                repeat(frustums[:, :1], "B ... -> (B2 B) ...", B2=B),
                chunk_size=chunk_size,
            )[:, 0],
            "(B B2) -> B B2",
            B=B,
        ).to(device)
        frustum_contained = frustum_contained | frustum_contained.T

    return frustum_contained
