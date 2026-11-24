"""Download and verify the native M7 chunks used by the published PHerc. 211 V9 benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from urllib.request import urlopen


def fetch(manifest, destination, workers):
    record = json.loads(manifest.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    base_url = record["native_array_url"].rstrip("/")
    chunks = record["chunks"]
    def copy_chunk(item):
        key, expected = item
        if not all(component.isdecimal() for component in key.split("/")):
            raise ValueError(f"Invalid chunk index {key!r}")
        file = destination / key
        if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == expected:
            return file.stat().st_size
        file.parent.mkdir(exist_ok=True, parents=True)
        for attempt in range(3):
            try:
                with urlopen(base_url + "/" + key, timeout=90) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Source checksum differs for {key}")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        with NamedTemporaryFile(dir=file.parent, delete=False) as temporary:
            temporary.write(data)
            pending = Path(temporary.name)
        try:
            pending.replace(file)
        finally:
            pending.unlink(missing_ok=True)
        return len(data)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        total = sum(executor.map(copy_chunk, chunks.items()))
    (destination / ".zarray").write_text(json.dumps(record["zarray"], indent=2) + "\n")
    print(f"Verified {len(chunks)} native Zarr chunks ({total:,} bytes) in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).parent / "benchmark" / "pherc211-v9-inputs.json")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    fetch(args.manifest, args.destination, args.workers)
