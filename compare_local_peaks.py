"""Compare local surface-peak snapping and two-sided tracking on the offline example."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from fill_sections import ZarrVolume, native_profiles, open_mesh
from verify_example import distances


def compare(example, reconstruction, evidence):
    metadata = json.loads((example / "volume.json").read_text())
    volume = ZarrVolume(example / "native_m7.zarr", metadata["origin_zyx"])
    restored, supported, _ = open_mesh(reconstruction)
    with np.load(evidence) as data:
        guided = data["guide"]
        normals = data["normal"]
        anchors = data["measured_anchors"]
    eligible = supported & ~anchors & np.isfinite(normals).all(axis=-1)
    depths = np.arange(-48, 49, dtype=np.float32)
    rays = native_profiles(volume, guided, normals, eligible, depths)
    smooth = ndimage.gaussian_filter1d(rays.astype(np.float32), 1.25, axis=-1)
    correction = np.zeros(eligible.shape, dtype=np.float32)
    for y, x in np.argwhere(eligible):
        peaks, _ = signal.find_peaks(smooth[y, x], height=65, prominence=20, distance=4)
        if peaks.size:
            nearest = peaks[np.argmin(abs(depths[peaks]))]
            correction[y, x] = depths[nearest]
    snapped = guided + normals * correction[..., None]
    reference, known, _ = open_mesh(example / "published-reference.tifxyz")
    common = eligible & known
    results = {}
    for name, positions in [("cubic", guided), ("nearest_peak", snapped), ("two_ended", restored)]:
        error = distances(positions, reference, known, common)
        error = error[np.isfinite(error)]
        results[name] = {
            "count": int(error.size),
            "within_four_voxels": float(np.mean(error <= 4)),
            "median_distance_voxels": float(np.median(error)),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.example, args.reconstruction, args.evidence), indent=2))
