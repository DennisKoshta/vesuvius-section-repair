"""Reconstruct the small PHerc. 211 example from two checksum-pinned public sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import zipfile

import numpy as np
from numcodecs import Blosc
import zarr

from fetch_benchmark_inputs import fetch
from fill_sections import open_mesh, write_mesh

HERE = Path(__file__).resolve().parent
MESH_URL = "https://drive.usercontent.google.com/download?id=1CGqmGKvKznjKCXw1xmo_OPclnLn681D3&export=download&confirm=t"
MESH_SHA256 = "9e9105d9bb812e6d2af454b30cb724e6339f0f0403e06eb28a302b484d31db36"
NATIVE_M7 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr/0/"


def prepare(destination, cache, supplied_archive, workers):
    if destination.exists():
        raise FileExistsError(f"Use a new destination or reuse the prepared example at {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    archive = supplied_archive or cache / "published-pherc211-v6-line2-mesh.zip"
    if supplied_archive is None and not archive.is_file():
        with TemporaryDirectory(dir=cache) as temporary:
            received = Path(temporary) / "mesh.zip"
            with urlopen(MESH_URL, timeout=100) as response, received.open("wb") as output:
                shutil.copyfileobj(response, output)
            with received.open("rb") as raw:
                if hashlib.file_digest(raw, "sha256").hexdigest() != MESH_SHA256:
                    raise ValueError("Public V6 download differs from the selected release.")
            received.replace(archive)
    with archive.open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != MESH_SHA256:
            raise ValueError("Public V6 mesh archive checksum mismatch.")
    manifest = HERE / "example" / "pherc211-native-inputs.json"
    if json.loads(manifest.read_text())["native_array_url"] != NATIVE_M7:
        raise ValueError("This example uses the designated PHerc. 211 native acquisition.")
    whole = cache / "public-pherc211-m7-level0.zarr"
    fetch(manifest, whole, workers)
    native = zarr.open_array(str(whole), mode="r")
    with TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary)
        parent = root / "parent.tifxyz"
        parent.mkdir()
        with zipfile.ZipFile(archive) as packed:
            for filename in ["x.tif", "y.tif", "z.tif", "meta.json"]:
                (parent / filename).write_bytes(packed.read("mesh/parent/normalized-tifxyz/" + filename))
        xyz, valid, metadata = open_mesh(parent)
        info = json.loads((HERE / "example" / "volume.json").read_text())
        y, x = info["reference_crop_chart_row_col_in_parent"]
        fixed = np.s_[y:y + 17, x:x + 14]
        xyz, valid = xyz[fixed], valid[fixed]
        anchors = np.zeros_like(valid)
        anchors[::8] = valid[::8]
        output = root / "prepared"
        output.mkdir()
        write_mesh(output / "published-reference.tifxyz", xyz, valid, metadata)
        write_mesh(output / "input.tifxyz", xyz, anchors, metadata)
        lower = np.asarray(info["origin_zyx"])
        upper = np.asarray(info["upper_bound_zyx"])
        shape = tuple(upper - lower)
        crop = zarr.open_array(str(output / "native_m7.zarr"), mode="w", zarr_format=2,
                               shape=shape, chunks=(96, 96, 96), dtype="u1", fill_value=0,
                               compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
        for z in range(0, shape[0], 96):
            for y in range(0, shape[1], 96):
                for x in range(0, shape[2], 96):
                    start = np.array([z, y, x])
                    stop = np.minimum(start + 96, shape)
                    target = tuple(slice(a, b) for a, b in zip(start, stop))
                    source = tuple(slice(a, b) for a, b in zip(lower + start, lower + stop))
                    values = native[source]
                    if values.any():
                        crop[target] = values
        crop.attrs["native_origin_zyx"] = lower.tolist()
        for filename in ["expected.json", "volume.json"]:
            shutil.copyfile(HERE / "example" / filename, output / filename)
        output.replace(destination)
    print(f"Prepared {int(anchors.sum())} measured PHerc. 211 vertices in {destination}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=Path("example-data"))
    parser.add_argument("--cache", type=Path, default=Path("public-input-cache"))
    parser.add_argument("--mesh-archive", type=Path, help="Use an existing V6 line-two public mesh ZIP.")
    parser.add_argument("--workers", default=4, type=int, choices=range(1,17))
    args = parser.parse_args()
    prepare(args.destination, args.cache, args.mesh_archive, args.workers)
