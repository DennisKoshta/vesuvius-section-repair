"""Reconstruct missing TIFXYZ mesh rows between measured papyrus sections."""

import argparse
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from scipy import ndimage
import tifffile


def open_mesh(path):
    path = Path(path)
    xyz = np.stack(
        [tifffile.imread(path / (a + ".tif")) for a in "zyx"], axis=-1
    ).astype("f")
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("Input coordinates must be three matching 2D TIFF images")
    ok = np.all(np.isfinite(xyz) & (xyz >= 0), axis=-1) & np.any(xyz != 0, axis=-1)
    if (path / "mask.tif").is_file():
        ok &= tifffile.imread(path / "mask.tif") > 0
    return xyz, ok, json.loads((path / "meta.json").read_text())


def write_mesh(path, xyz, ok, meta):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for k, a in enumerate("zyx"):
        tifffile.imwrite(
            path / (a + ".tif"), np.where(ok, xyz[:, :, k], -1).astype("f")
        )
    tifffile.imwrite(path / "mask.tif", np.uint8(ok) * 255)
    out = dict(meta)
    out["type"] = "seg"
    out["format"] = "tifxyz"
    out["uuid"] = path.stem
    if ok.any():
        p = xyz[ok][:, ::-1]
        out["bbox"] = [p.min(0).tolist(), p.max(0).tolist()]
    for key in ("area_cm2", "area_vx2"):
        out.pop(key, None)
    (path / "meta.json").write_text(json.dumps(out, indent=2))


@lru_cache(maxsize=1)
def load_kernel():
    source = Path(__file__).with_name("native_match.cpp")
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / "native_match.so"
        subprocess.run(
            ["g++", "-std=c++17", "-O3", "-fPIC", "-fopenmp", "-shared",
             str(source), "-o", str(library)],
            check=True,
        )
        fn = ctypes.CDLL(str(library)).transport_brackets
    p, i = ctypes.c_void_p, ctypes.c_int
    fn.argtypes = [p, i, i, i, p, i, i, i, p, p, p]
    fn.restype = None
    return fn


class ZarrVolume:
    def __init__(self, path, origin):
        import zarr

        if "://" in str(path):
            import fsspec

            options = {"client_kwargs": {"trust_env": True}} if str(path).startswith(("http://", "https://")) else {}
            array = zarr.open(fsspec.get_mapper(path, **options), mode="r")
        else:
            array = zarr.open(str(path), mode="r")
        if not hasattr(array, "shape"):
            array = array["0"]
        if len(array.shape) != 3 or np.dtype(array.dtype) != np.dtype("uint8"):
            raise ValueError(
                "The surface-prediction input must be a native z,y,x uint8 Zarr array"
            )
        self.array = array
        self.origin = np.asarray(origin, int)
        self.low = self.origin
        self.high = self.origin + np.asarray(array.shape)

    def window(self, lo, hi):
        lower = np.asarray(lo) - self.origin
        upper = np.asarray(hi) - self.origin
        return np.asarray(
            self.array[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
        )


def column_tangents(xyz, valid):
    tangent = np.full_like(xyz, np.nan)
    left = np.roll(valid, 1, axis=1)
    right = np.roll(valid, -1, axis=1)
    left[:, 0] = False
    right[:, -1] = False
    back = xyz - np.roll(xyz, 1, axis=1)
    ahead = np.roll(xyz, -1, axis=1) - xyz
    tangent[valid & left] = back[valid & left]
    tangent[valid & right] = ahead[valid & right]
    both = valid & left & right
    tangent[both] = (back[both] + ahead[both]) / 2
    return tangent


def gaps(valid, max_gap):
    records = {}
    for col in range(valid.shape[1]):
        rows = np.flatnonzero(valid[:, col])
        for k in range(len(rows) - 1):
            lo, hi = map(int, rows[k : k + 2])
            if hi - lo < 2 or hi - lo > max_gap:
                continue
            left = int(rows[k - 1]) if k else -1
            right = int(rows[k + 2]) if k + 2 < len(rows) else -1
            records.setdefault((lo, hi), []).append((col, left, right))
    return records


def section_geometry(xyz, tangents, lo, hi, columns, left, right, steps):
    t = np.linspace(0.0, 1.0, steps + 1, dtype="f")[:, None, None]
    A, B = xyz[lo, columns], xyz[hi, columns]
    gap = hi - lo
    chord = B - A
    m0 = chord.copy()
    m1 = chord.copy()
    p = left >= 0
    s = right >= 0
    if p.any():
        m0[p] = (B[p] - xyz[left[p], columns[p]]) * (gap / (hi - left[p]))[:, None]
    if s.any():
        m1[s] = (xyz[right[s], columns[s]] - A[s]) * (gap / (right[s] - lo))[:, None]
    center = (
        (2 * t**3 - 3 * t * t + 1) * A
        + (t**3 - 2 * t * t + t) * m0
        + (-2 * t**3 + 3 * t * t) * B
        + (t**3 - t * t) * m1
    )
    dv = (
        (6 * t * t - 6 * t) * A
        + (3 * t * t - 4 * t + 1) * m0
        + (-6 * t * t + 6 * t) * B
        + (3 * t * t - 2 * t) * m1
    )
    du = (1 - t) * tangents[lo, columns] + t * tangents[hi, columns]
    n = np.cross(du, dv)
    length = np.linalg.norm(n, axis=-1)
    ok = np.isfinite(length) & (length > 1.0e-4)
    n /= np.maximum(length[:, :, None], 1.0e-7)
    return center.astype("f"), n.astype("f"), ok


def native_profiles(volume, pos, norm, valid, depth):
    T, W = valid.shape
    raw = np.zeros((T, W, len(depth)), "u1")
    for i in range(0, T, 7):
        for j in range(0, W, 12):
            p = pos[i : i + 7, j : j + 12]
            n = norm[i : i + 7, j : j + 12]
            ok = valid[i : i + 7, j : j + 12]
            if not ok.any():
                continue
            samples = (
                p[ok, :, None] + n[ok, :, None] * depth[None, None, :]
            ).transpose(1, 0, 2)
            real = np.all(
                (samples >= volume.low[:, None, None] + 1)
                & (samples < volume.high[:, None, None] - 2),
                axis=(0, 2),
            )
            if not real.all():
                selection = np.argwhere(ok)
                bad = selection[~real]
                ok[bad[:, 0], bad[:, 1]] = False
                samples = samples[:, real]
            if not ok.any():
                continue
            lower = np.floor(samples.min((1, 2))).astype(int)
            upper = np.ceil(samples.max((1, 2))).astype(int) + 1
            # Refuse to read an unreasonably large bounding box across a folded
            # or disconnected part. Split into individual rays in this case.
            if np.prod(upper - lower) > 18_000_000:
                pts = []
                for ray in samples.transpose(1, 0, 2):
                    low = np.floor(ray.min(1)).astype(int)
                    high = np.ceil(ray.max(1)).astype(int) + 1
                    pts.append(
                        ndimage.map_coordinates(
                            volume.window(low, high),
                            ray - low[:, None],
                            order=1,
                            prefilter=False,
                        )
                    )
                vals = np.asarray(pts)
            else:
                vals = ndimage.map_coordinates(
                    volume.window(lower, upper),
                    samples - lower[:, None, None],
                    order=1,
                    prefilter=False,
                )
            raw[i : i + 7, j : j + 12][ok] = vals
    return np.ascontiguousarray(ndimage.gaussian_filter1d(raw, 1.25, axis=2))


def fit_profiles(raw, valid, fn):
    T, W, D = raw.shape
    anchors = np.full((T, W), np.nan, "f")
    anchors[0] = D // 2
    anchors[-1] = D // 2
    keep = valid.all(0)
    anchors[:, ~keep] = np.nan
    t = np.linspace(0, 1, T, dtype="f")[:, None]
    proposals = []
    closures = []
    qualities = []
    for half in (18, 33):
        f, b, q = [np.full((T, W), np.nan, "f") for _ in range(3)]
        fn(
            raw.ctypes.data,
            T,
            W,
            D,
            anchors.ctypes.data,
            T - 1,
            half,
            9,
            f.ctypes.data,
            b.ctypes.data,
            q.ctypes.data,
        )
        proposals.append((1 - t) * f + t * b - D // 2)
        closures.append(abs(f - b))
        qualities.append(q)
    delta = (proposals[0] + proposals[1]) / 2.0
    closure = np.maximum(*closures)
    agreement = abs(proposals[0] - proposals[1])
    quality = np.minimum(*qualities)
    trusted = (
        valid
        & (agreement <= 4)
        & (closure <= 8)
        & (quality >= 0.62)
        & (abs(delta) < D // 2 - 44)
    )
    return delta, trusted, closure, agreement, quality


def fill(xyz, valid, volume, max_gap=20, ray_half=80, fine_step=4.0, row_axis=0):
    if max_gap < 2 or ray_half < 50 or fine_step <= 0:
        raise ValueError("max-gap >= 2, ray-half >= 50 and fine-step > 0 are required")
    if row_axis == 1:
        xyz, valid = xyz.swapaxes(0, 1), valid.T
    target = xyz.copy()
    accepted = valid.copy()
    shape = valid.shape
    guide = np.full_like(xyz, np.nan)
    normal = np.full_like(xyz, np.nan)
    candidate = np.full_like(xyz, np.nan)
    diagnostics = {
        name: np.full(shape, np.nan, "f")
        for name in (
            "path_closure",
            "width_disagreement",
            "quality",
            "normal_displacement",
        )
    }
    tangents = column_tangents(xyz, valid)
    fn = load_kernel()
    depth = np.arange(-ray_half, ray_half + 1, dtype="f")
    empty = ~valid
    start = time.monotonic()
    for (lo, hi), rows in sorted(gaps(valid, max_gap).items()):
        records = np.array(rows, int)
        gap = hi - lo
        for offset in range(0, len(rows), 32):
            cols, left, right = records[offset : offset + 32].T
            speed = np.linalg.norm(xyz[hi, cols] - xyz[lo, cols], axis=1) / gap
            stride = int(np.clip(np.ceil(np.percentile(speed, 85) / fine_step), 2, 20))
            center, n, ok = section_geometry(
                xyz, tangents, lo, hi, cols, left, right, gap * stride
            )
            rays = native_profiles(volume, center, n, ok, depth)
            shift, fit, close, spread, qual = fit_profiles(rays, ok, fn)
            for k in range(1, gap):
                j = k * stride
                row = lo + k
                guide[row, cols] = center[j]
                normal[row, cols] = n[j]
                out = center[j] + shift[j, :, None] * n[j]
                candidate[row, cols] = out
                for key, a in zip(diagnostics, (close, spread, qual, shift)):
                    diagnostics[key][row, cols] = a[j]
                hits = fit[j] & empty[row, cols]
                target[row, cols[hits]] = out[hits]
                accepted[row, cols[hits]] = True
        print(
            "measured section rows",
            lo,
            hi,
            "candidate columns",
            len(records),
            "elapsed seconds",
            round(time.monotonic() - start, 1),
            flush=True,
        )
    arrays = dict(
        guide=guide,
        normal=normal,
        unfiltered_candidate=candidate,
        accepted_fill=accepted & ~valid,
        measured_anchors=valid,
        **diagnostics,
    )
    if row_axis == 1:
        target, accepted = target.swapaxes(0, 1), accepted.T
        arrays = {k: v.swapaxes(0, 1) for k, v in arrays.items()}
    return target, accepted, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh")
    p.add_argument("surface_zarr")
    p.add_argument("output")
    p.add_argument("--origin-zyx", type=int, nargs=3, default=(0, 0, 0))
    p.add_argument("--max-gap", type=int, default=20)
    p.add_argument("--ray-half", type=int, default=80)
    p.add_argument("--fine-step", type=float, default=4)
    p.add_argument("--axis", choices=["rows", "columns"], default="rows")
    p.add_argument("--evidence")
    a = p.parse_args()
    if Path(a.mesh).resolve() == Path(a.output).resolve():
        raise ValueError("Use a distinct output TIFXYZ directory")
    if a.max_gap < 2:
        p.error("--max-gap must allow at least one missing vertex")
    if a.ray_half < 46:
        p.error("--ray-half must leave room for both profile windows and their search")
    if not np.isfinite(a.fine_step) or a.fine_step <= 0:
        p.error("--fine-step must be positive and finite")
    if Path(a.output).exists() and not Path(a.output).is_dir():
        p.error("The output path must name a mesh directory")
    xyz, ok, meta = open_mesh(a.mesh)
    volume = ZarrVolume(a.surface_zarr, a.origin_zyx)
    target, valid, records = fill(
        xyz,
        ok,
        volume,
        a.max_gap,
        a.ray_half,
        a.fine_step,
        0 if a.axis == "rows" else 1,
    )
    if not np.array_equal(target[ok], xyz[ok]) or not valid[ok].all():
        raise RuntimeError("Input anchor preservation failed")
    write_mesh(a.output, target, valid, meta)
    if a.evidence:
        np.savez_compressed(a.evidence, **records)
    print(
        json.dumps(
            {
                "measured_input_vertices": int(ok.sum()),
                "recovered_vertices": int((valid & ~ok).sum()),
                "output": a.output,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
