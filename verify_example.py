"""Verify the offline PHerc. 211 reconstruction against the withheld released mesh."""

import argparse
import json
from pathlib import Path
import numpy as np
from fill_sections import open_mesh


def triangle_distance(p, a, b, c):
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", normal, normal)
    signed = np.einsum("ij,ij->i", p - a, normal) / np.maximum(nn, 1e-12)
    proj = p - signed[:, None] * normal
    d0 = np.einsum("ij,ij->i", ab, ab)
    d1 = np.einsum("ij,ij->i", ab, ac)
    d2 = np.einsum("ij,ij->i", ac, ac)
    v = proj - a
    v1 = np.einsum("ij,ij->i", v, ab)
    v2 = np.einsum("ij,ij->i", v, ac)
    den = d0 * d2 - d1 * d1
    u = (d2 * v1 - d1 * v2) / np.maximum(den, 1e-9)
    w = (d0 * v2 - d1 * v1) / np.maximum(den, 1e-9)
    inside = (den > 1e-7) & (u >= 0) & (w >= 0) & (u + w <= 1.0)
    result = np.where(inside, signed * signed * nn, np.inf)
    for s, t in ((a, b), (b, c), (c, a)):
        e = t - s
        pn = p - s
        coef = np.clip(
            np.einsum("ij,ij->i", pn, e)
            / np.maximum(np.einsum("ij,ij->i", e, e), 1e-12),
            0,
            1,
        )
        d = pn - coef[:, None] * e
        result = np.minimum(result, np.einsum("ij,ij->i", d, d))
    return result


def distances(pred, xyz, valid, where):
    i, j = np.where(where)
    p = pred[i, j]
    d = np.full(len(p), np.inf)
    for dy in range(-2, 2):
        for dx in range(-2, 2):
            y = i + dy
            x = j + dx
            ok = (
                (y >= 0)
                & (x >= 0)
                & (y + 1 < valid.shape[0])
                & (x + 1 < valid.shape[1])
            )
            y = y.clip(0, valid.shape[0] - 2)
            x = x.clip(0, valid.shape[1] - 2)
            ok &= valid[y, x] & valid[y + 1, x] & valid[y, x + 1] & valid[y + 1, x + 1]
            if not ok.any():
                continue
            iy, ix = y[ok], x[ok]
            a, b, c, e = (
                xyz[iy, ix],
                xyz[iy + 1, ix],
                xyz[iy, ix + 1],
                xyz[iy + 1, ix + 1],
            )
            d[ok] = np.minimum(
                d[ok],
                np.minimum(
                    triangle_distance(p[ok], a, b, e), triangle_distance(p[ok], a, c, e)
                ),
            )
    ans = np.full(valid.shape, np.nan, "f")
    ans[i, j] = np.where(np.isfinite(d), np.sqrt(d), np.nan)
    return ans


def stats(a, use):
    v = a[use & np.isfinite(a)]
    return {
        "n": len(v),
        "within4": float(np.mean(v <= 4)) if len(v) else None,
        "within8": float(np.mean(v <= 8)) if len(v) else None,
        "median_p90_p99": np.percentile(v, [50, 90, 99]).tolist() if len(v) else None,
    }


def verify(example, result, evidence):
    sparse, anchors, _ = open_mesh(example / "input.tifxyz")
    released, valid, _ = open_mesh(example / "published-reference.tifxyz")
    recovered, supported, _ = open_mesh(result)
    with np.load(evidence) as archive:
        guide = archive["guide"]
    if not np.array_equal(recovered[anchors], sparse[anchors]):
        raise ValueError("The recovered mesh changed at least one measured anchor")
    check = valid & supported & ~anchors
    report = {}
    for name, prediction in (("cubic", guide), ("repaired", recovered)):
        dist = distances(prediction, released, valid, check)
        report[name] = stats(dist, check)
    expected = json.loads((example / "expected.json").read_text())["on_hidden_vertices"]
    for method, reference in (("cubic", "cubic"), ("repaired", "tracked")):
        if report[method]["n"] != expected[reference]["n"]:
            raise ValueError(
                "Comparable vertex count differs from the supplied reference"
            )
        if abs(report[method]["within4"] - expected[reference]["within4"]) > 1e-7:
            raise ValueError("Four-voxel fidelity differs from the supplied reference")
        if not np.allclose(
            report[method]["median_p90_p99"],
            expected[reference]["median_p90_p99"],
            rtol=2e-4,
            atol=0.004,
        ):
            raise ValueError("Quantile distances differ from the supplied reference")
    report["preserved_input_vertices"] = int(anchors.sum())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("example", type=Path)
    p.add_argument("reconstruction", type=Path)
    p.add_argument("evidence", type=Path)
    a = p.parse_args()
    verify(a.example, a.reconstruction, a.evidence)
