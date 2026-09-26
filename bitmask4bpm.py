import os
import glob
import re
import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clipped_stats

SALT_PEPPER = np.uint8(1)
HIGH = np.uint8(2)
LOW = np.uint8(4)
STUCK = np.uint8(8)
BLINKING = np.uint8(16)

BITS = {
    "salt_and_pepper": SALT_PEPPER,
    "high": HIGH,
    "low": LOW,
    "stuck": STUCK,
    "blinking": BLINKING,
}



def make_bitmask(img_paths, output_path, target_n=None):
    img_paths = sorted(img_paths)
    img_paths, bad_images = keep_good_images(img_paths)

    if target_n is not None:
        img_paths = img_paths[:target_n]

    if len(img_paths) < 4:
        raise ValueError("At least 4 usable frames are required")

    img_shape, nx, ny = camera_settings(img_paths[0])
    camera_name = os.path.basename(img_paths[0]).split("_202")[0]
    maxp = max_percentile(nx, ny, camera_name)

    bitmask = np.zeros(img_shape, dtype=np.uint8)
    n = len(img_paths)
    wx = img_shape[1] // nx
    hy = img_shape[0] // ny

    for j in range(ny):
        for k in range(nx):
            cube = np.empty((hy, wx, n), dtype=np.float32)

            for i, path in enumerate(img_paths):
                image = load_image(path)
                cube[:, :, i] = image[
                    j * hy:(j + 1) * hy,
                    k * wx:(k + 1) * wx
                ]

            flat = cube.reshape(hy * wx, n)

            filtered = ndimage.median_filter(cube, size=(3, 3, 1))
            diff = np.abs(cube - filtered).reshape(hy * wx, n)
            threshold = 6 * np.std(cube)
            sandp = np.all(diff > threshold, axis=1)

            frame_median = np.nanmedian(cube, axis=(0, 1))
            frame_std = np.nanstd(cube, axis=(0, 1))

            high_count = np.sum(
                cube > frame_median[None, None, :] + 6 * frame_std[None, None, :],
                axis=2
            )
            low_count = np.sum(
                cube < frame_median[None, None, :] - 6 * frame_std[None, None, :],
                axis=2
            )

            for i in range(n):
                _, med, std = sigma_clipped_stats(cube[:, :, i])
                cube[:, :, i] = np.clip(
                    cube[:, :, i],
                    med - 3 * std,
                    med + 3 * std
                )

            median_stack = np.nanmedian(cube, axis=2)
            global_median = np.nanmedian(median_stack)
            global_std = np.nanstd(median_stack)

            high = (
                (median_stack > global_median + 6 * global_std)
                & (high_count >= max(2, int(0.2 * n)))
            ).reshape(hy * wx)

            low = (
                (median_stack < global_median - 6 * global_std)
                & (low_count >= int(0.2 * n))
            ).reshape(hy * wx)

            flat = cube.reshape(hy * wx, n)

            stuck = np.std(flat, axis=1) < 1e-6

            std1 = np.nanstd(flat[:, :n // 2], axis=1)
            std2 = np.std(flat[:, n // 2:], axis=1)
            p1 = np.percentile(std1, maxp)
            p2 = np.percentile(std2, maxp)
            blinking = (std1 > p1) & (std2 > p2)
