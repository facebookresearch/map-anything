"""Regression tests for camera geometry under integer-sized image resizes."""

import cv2
import numpy as np
import pytest
from PIL import Image

from mapanything.utils.cropping import (
    crop_resize_if_necessary,
    rescale_image_and_other_optional_info,
)


@pytest.mark.parametrize(
    "source_wh,target_wh",
    [
        ((103, 81), (50, 40)),
        ((81, 103), (40, 50)),
        ((1280, 720), (924, 518)),
        ((720, 1280), (518, 924)),
        ((63, 31), (100, 60)),
        ((31, 63), (60, 100)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_intrinsics_match_realized_resize(source_wh, target_wh, dtype):
    width, height = source_wh
    intrinsics = np.array(
        [[100, 3, width * 0.43], [0, 110, height * 0.61], [0, 0, 1]],
        dtype=dtype,
    )
    original = intrinsics.copy()
    image, _, resized_intrinsics, _ = rescale_image_and_other_optional_info(
        Image.new("RGB", source_wh), target_wh, camera_intrinsics=intrinsics
    )
    scale_xy = np.array([image.width / width, image.height / height])
    expected = intrinsics.copy()
    expected[:2, :] *= scale_xy[:, None]
    expected[:2, 2] += (scale_xy - 1) / 2
    np.testing.assert_allclose(resized_intrinsics, expected, rtol=2e-7, atol=1e-5)
    np.testing.assert_array_equal(intrinsics, original)
    assert resized_intrinsics.dtype == dtype
    assert not np.shares_memory(intrinsics, resized_intrinsics)


@pytest.mark.parametrize("source_wh", [(103, 81), (81, 103)])
def test_crop_resize_matches_sampled_coordinates(source_wh):
    width, height = source_wh
    target_wh = (50, 40) if width > height else (40, 50)
    intrinsics = np.array(
        [[100.0, 0, width * 0.43], [0, 110.0, height * 0.61], [0, 0, 1]]
    )
    for axis in (0, 1):
        coordinates = np.indices((height, width), dtype=np.float32)[1 - axis]
        image, resized_intrinsics = crop_resize_if_necessary(
            Image.fromarray(coordinates), target_wh, intrinsics=intrinsics
        )
        assert image.size == target_wh
        raster = np.asarray(image)
        for fraction in (0.25, 0.75):
            column = int((image.width - 1) * fraction)
            row = int((image.height - 1) * fraction)
            output_coordinate = (column, row)[axis]
            source_coordinate = (
                output_coordinate - resized_intrinsics[axis, 2]
            ) * intrinsics[axis, axis] / resized_intrinsics[axis, axis] + intrinsics[
                axis, 2
            ]
            assert abs(source_coordinate - raster[row, column]) < 0.03


@pytest.mark.parametrize("source_wh", [(103, 81), (81, 103)])
def test_resize_keeps_existing_raster_depth_and_auxiliary_sampling(source_wh):
    width, height = source_wh
    target_wh = (50, 40) if width > height else (40, 50)
    image = np.random.default_rng(1).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    depth = np.arange(height * width, dtype=np.float32).reshape(height, width)
    mask = (depth % 3 == 0).astype(np.uint8)
    original_image, original_depth, original_mask = (
        array.copy() for array in (image, depth, mask)
    )
    resized, resized_depth, intrinsics, auxiliary = (
        rescale_image_and_other_optional_info(
            image,
            target_wh,
            depthmap=depth,
            additional_quantities_to_be_resized_with_nearest=[mask],
        )
    )
    scale = max(np.array(target_wh) / source_wh) + 1e-8
    resized_wh = tuple(np.floor(np.array(source_wh) * scale).astype(int))
    assert resized.size == resized_wh
    assert resized_depth.shape == auxiliary[0].shape == resized_wh[::-1]
    expected_image = Image.fromarray(image).resize(
        resized_wh, resample=Image.Resampling.LANCZOS
    )
    np.testing.assert_array_equal(np.asarray(resized), np.asarray(expected_image))
    for source, actual in ((depth, resized_depth), (mask, auxiliary[0])):
        np.testing.assert_array_equal(
            actual, cv2.resize(source, resized_wh, interpolation=cv2.INTER_NEAREST)
        )
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(depth, original_depth)
    np.testing.assert_array_equal(mask, original_mask)
    assert intrinsics is None


@pytest.mark.parametrize("target_wh", [(32, 16), (64, 32), (128, 64)])
def test_exact_and_identity_resizes_preserve_pixel_centers(target_wh):
    source_wh = (64, 32)
    intrinsics = np.array([[64.0, 0, 31.5], [0, 32.0, 15.5], [0, 0, 1]])
    image, _, resized_intrinsics, _ = rescale_image_and_other_optional_info(
        Image.new("RGB", source_wh), target_wh, camera_intrinsics=intrinsics
    )
    np.testing.assert_allclose(
        resized_intrinsics[:2, 2],
        ((image.width - 1) / 2, (image.height - 1) / 2),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        resized_intrinsics[[0, 1], [0, 1]], image.size, rtol=0, atol=1e-12
    )


def test_force_false_preserves_original_objects():
    image = Image.new("RGB", (32, 16))
    depth = np.ones((16, 32), dtype=np.float32)
    intrinsics = np.eye(3)
    auxiliary = [np.ones((16, 32), dtype=np.uint8)]
    output = rescale_image_and_other_optional_info(
        image,
        (64, 32),
        depthmap=depth,
        camera_intrinsics=intrinsics,
        force=False,
        additional_quantities_to_be_resized_with_nearest=auxiliary,
    )
    assert all(
        actual is original
        for actual, original in zip(output, (image, depth, intrinsics, auxiliary))
    )
