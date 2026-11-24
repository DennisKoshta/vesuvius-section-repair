"""Draw the measured PHerc. 211 axial CT section and the three registered geometry traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import zarr

from fetch_benchmark_inputs import fetch
from fill_sections import open_mesh

HERE = Path(__file__).resolve().parent


def segments(mesh, valid, axial):
    faces = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    a, b, c, d = mesh[:-1, :-1], mesh[1:, :-1], mesh[:-1, 1:], mesh[1:, 1:]
    for triangle in np.concatenate([np.stack([a, b, d], axis=-2)[faces],
                                    np.stack([a, d, c], axis=-2)[faces]]):
        points = []
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            v, w = triangle[i], triangle[j]
            if (v[0] <= axial < w[0]) or (w[0] <= axial < v[0]):
                p = v + (axial - v[0]) / (w[0] - v[0]) * (w - v)
                points.append(p[1:])
        if len(points) == 2:
            yield points


def render(example, reconstructed, evidence, output, cache):
    manifest = HERE / "example" / "pherc211-ct-section-inputs.json"
    source = json.loads(manifest.read_text())
    raw_root = cache / "public-pherc211-ct-level0.zarr"
    fetch(manifest, raw_root, 4)
    volume = zarr.open_array(str(raw_root), mode="r")
    z, y, x = source["native_center_zyx"]
    half = source["half_window_voxels"]
    raw = volume[z, y-half:y+half+1, x-half:x+half+1]
    lower, upper = np.percentile(raw, [1, 99])
    gray = np.uint8(255 * np.clip((raw.astype(float)-lower)/max(upper-lower, 1), 0, 1))
    base = Image.fromarray(gray).convert("RGB")
    repaired, accepted, _ = open_mesh(reconstructed)
    reference, known, _ = open_mesh(example / "published-reference.tifxyz")
    with np.load(evidence) as fitted:
        cubic = fitted["guide"]
    common = accepted & known & np.isfinite(cubic).all(-1)
    panels = [("Native CT", base)]
    for label, mesh in [("Cubic interpolation", cubic), ("Two-ended reconstruction", repaired),
                        ("Withheld published surface", reference)]:
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        for endpoints in segments(mesh, common, z):
            line = [(float(pt[1]-x+half), float(pt[0]-y+half)) for pt in endpoints]
            draw.line(line, fill=(253, 113, 62), width=1)
        panels.append((label, panel))
    size = 265
    figure = Image.new("RGB", (len(panels)*size, size+45), (25, 34, 39))
    labels = ImageDraw.Draw(figure)
    for i, (title, img) in enumerate(panels):
        labels.text((i*size+5, 5), title, fill="white")
        figure.paste(img.resize((size-4, size-4), Image.Resampling.NEAREST), (size*i+2, 23))
    labels.text((5, size+25), f"PHerc. 211, native axial z {z}. All panels show identical raw voxels and image scaling.",
                fill="white")
    figure.save(output)
    print(f"Saved the registered native cross-section to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("public-input-cache"))
    args = parser.parse_args()
    render(args.example, args.reconstruction, args.evidence, args.output, args.cache)
