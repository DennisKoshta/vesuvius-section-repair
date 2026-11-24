"""Withhold measured sections of a released TIFXYZ mesh and score a blind reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from fill_sections import ZarrVolume, fill, open_mesh, write_mesh
from verify_example import distances, stats


def profile_samples(volume, points, normals, depths):
    """Sample unsmoothed native values at registered ray coordinates."""
    groups, group_ids = np.unique(np.floor(points / 96).astype(int), axis=0, return_inverse=True)
    order = np.argsort(group_ids, kind="stable")
    boundaries = np.r_[0, np.cumsum(np.bincount(group_ids))]
    rays = np.zeros((len(points), len(depths)), dtype=np.uint8)
    for group in range(len(groups)):
        ids = order[boundaries[group]:boundaries[group + 1]]
        q = (points[ids, :, None] + normals[ids, :, None] * depths).transpose(1, 0, 2)
        low = np.maximum(np.floor(q.min(axis=(1, 2))).astype(int), volume.low)
        high = np.minimum(np.ceil(q.max(axis=(1, 2))).astype(int) + 1, volume.high)
        if np.all(high > low):
            rays[ids] = ndimage.map_coordinates(volume.window(low, high), q - low[:, None, None],
                                                order=1, mode="constant", prefilter=False)
    return rays


def local_baselines(volume, guide, normals, eligible, anchors):
    iy, ix = np.where(eligible)
    depths = np.arange(-48, 49, dtype=np.float32)
    raw = profile_samples(volume, guide[iy, ix], normals[iy, ix], depths)
    smooth = ndimage.gaussian_filter1d(raw.astype(np.float32), 1.25, axis=1)
    nearest = np.zeros(len(raw), dtype=np.float32)
    weighted = nearest.copy()
    all_peaks = []
    for i, profile in enumerate(smooth):
        peaks, _ = signal.find_peaks(profile, height=65, prominence=20, distance=4)
        all_peaks.append(peaks)
        if peaks.size:
            nearest[i] = depths[peaks[np.argmin(abs(depths[peaks]))]]
            score = profile[peaks] / 255.0 - 0.11 * (depths[peaks] / 12.0) ** 2
            weighted[i] = depths[peaks[np.argmax(score)]]
    neighbors = np.full(anchors.shape, np.nan, dtype=np.float32)
    neighbors[anchors] = 0
    neighbors[iy, ix] = nearest
    kernel = np.ones((3, 3), dtype=np.float32)
    counts = ndimage.convolve(np.isfinite(neighbors).astype(np.float32), kernel, mode="constant")
    totals = ndimage.convolve(np.nan_to_num(neighbors), kernel, mode="constant")
    means = (totals / np.maximum(counts, 1))[iy, ix]
    guided = nearest.copy()
    for i, peaks in enumerate(all_peaks):
        if peaks.size:
            guided[i] = depths[peaks[np.argmin(abs(depths[peaks] - means[i]))]]
    inner = abs(depths) <= 12
    shifts = {"nearest-prominent-peak": nearest, "neighbor-guided-nearest-peak": guided,
              "prominent-peak-with-distance-prior": weighted,
              "brightest-within-12": depths[inner][np.argmax(smooth[:, inner], axis=1)]}
    for name, offsets in shifts.items():
        candidate = guide.copy()
        candidate[iy, ix] += normals[iy, ix] * offsets[:, None]
        yield name, candidate


def benchmark(reference, surface_zarr, destination, gap, axis, origin, peaks):
    if gap < 2:
        raise ValueError("The withheld-section test needs a gap of two or greater")
    destination.mkdir(parents=True, exist_ok=False)
    full, valid, metadata = open_mesh(reference)
    anchors = np.zeros_like(valid)
    if axis == "rows":
        anchors[::gap] = valid[::gap]
    else:
        anchors[:, ::gap] = valid[:, ::gap]
    sparse_path = destination / "sparse-input.tifxyz"
    write_mesh(sparse_path, full, anchors, metadata)
    points, measured, metadata = open_mesh(sparse_path)
    volume = ZarrVolume(surface_zarr, origin)
    recovered, supported, evidence = fill(points, measured, volume, max_gap=gap, ray_half=78,
                                          fine_step=4.0, row_axis=int(axis == "columns"))
    if not np.array_equal(recovered[measured], points[measured]) or not supported[measured].all():
        raise ValueError("A reconstruction changed an original measured anchor")
    write_mesh(destination / "repaired.tifxyz", recovered, supported, metadata)
    np.savez_compressed(destination / "evidence.npz", **evidence)
    guide = evidence["guide"]
    tracked = evidence["unfiltered_candidate"]
    eligible = ~measured & np.isfinite(guide).all(-1) & np.isfinite(tracked).all(-1)
    accepted = evidence["accepted_fill"]
    comparable = eligible & valid
    metrics = {}
    def score(name, candidate):
        error = distances(candidate, full, valid, comparable)
        metrics[name] = {"all_with_reference": stats(error, comparable),
                         "on_accepted_fill": stats(error, comparable & accepted)}
    score("cubic_guide", guide)
    score("tracked", tracked)
    if peaks and eligible.any():
        for name, candidate in local_baselines(volume, guide, evidence["normal"], eligible, measured):
            score(name, candidate)
    report = {"axis": axis, "anchor_gap": gap, "shape": list(valid.shape),
              "hidden_valid": int((valid & ~measured).sum()), "anchors_preserved": int(measured.sum()),
              "accepted_hidden": int((valid & accepted).sum()), "metrics": metrics}
    (destination / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("surface_zarr")
    parser.add_argument("new_output_directory", type=Path)
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--axis", choices=["rows", "columns"], default="rows")
    parser.add_argument("--origin-zyx", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--local-peaks", action="store_true")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.reference, args.surface_zarr, args.new_output_directory,
                               args.gap, args.axis, args.origin_zyx, args.local_peaks), indent=2))
