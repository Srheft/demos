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

def load_image(path):
    return np.squeeze(np.load(path)["arr_0"]).astype(np.float32)


def camera_settings(path):
    name = os.path.basename(path)
    if "nsv455" in name:
        return (6388, 9576), 3, 4
    if "nsv571" in name:
        return (4134, 6120), 2, 2
    raise ValueError(name)
    

def max_percentile(nx, ny, camera_name):
    if "nsv455" in camera_name:
        return min(99.99999, 99.99 + 20 * (nx * ny) / 32 * 0.0005)
    return min(99.99999, 99.99 + 20 * (nx * ny) / 16 * 0.0005)


def keep_good_images(paths):
    keep = []
    bad = []
    for path in paths:
        image = load_image(path)
        if np.mean(image) > 5000 or np.sum(image) < 1:
            bad.append(path)
        else:
            keep.append(path)
    return keep, bad

########################################################################

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

            tile = np.zeros(hy * wx, dtype=np.uint8)
            tile[sandp] |= SALT_PEPPER
            tile[high] |= HIGH
            tile[low] |= LOW
            tile[stuck] |= STUCK
            tile[blinking] |= BLINKING

            bitmask[
                j * hy:(j + 1) * hy,
                k * wx:(k + 1) * wx
            ] = tile.reshape(hy, wx)

    counts = np.array(
        [np.count_nonzero(bitmask & bit) for bit in BITS.values()],
        dtype=np.int64
    )

    names = np.array(list(BITS.keys()))
    bits = np.array(list(BITS.values()), dtype=np.uint8)
    bpm = bitmask == 0

    np.savez_compressed(
        output_path,
        bitmask=bitmask,
        bpm=bpm,
        names=names,
        bits=bits,
        counts=counts,
        n_frames=np.int64(n),
        bad_images=np.array(bad_images),
        input_images=np.array(img_paths),
    )

    return bitmask

####################################################################################################

def decode_pixel(bitmask, row, column):
    value = int(bitmask[row, column])
    categories = [
        name for name, bit in BITS.items()
        if value & int(bit)
    ]
    return value, categories


bitmask = make_bitmask(paths, output_path, target_n=target_n)

blinking = (bitmask & 16) != 0

