# Repairing missing papyrus mesh sections with native surface predictions

This CPU tool fills gaps between measured sections of a papyrus mesh. It follows the surrounding sheet structure from both ends of each gap, keeps supported vertices and writes a standard TIFXYZ mesh that can be opened in VC3D. The PHerc. 211 example uses a public surface and public native scan predictions. A one-time download prepares it for offline use.

## Run the example

Use a Linux environment with Python 3.12 and a C++17 compiler with OpenMP. The numerical run below was checked with the versions in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_example.py example-data
OMP_NUM_THREADS=2 python fill_sections.py example-data/input.tifxyz example-data/native_m7.zarr repaired.tifxyz \
  --origin-zyx 7444 3315 3198 --max-gap 8 --ray-half 78 --evidence fit.npz
python verify_example.py example-data repaired.tifxyz fit.npz
```

Preparation verifies Priyank Aranke’s published V6 line-two mesh archive and 32 native M7 chunks from the Vesuvius Challenge data warehouse. It extracts rows 121–137 and columns 231–244 of the parent mesh. Passing an existing file with `--mesh-archive` reuses the downloaded archive. The repair and verification commands both run offline once `example-data` exists.

The input has 42 vertices on three measured rows. In this example 192 hidden vertices can be checked against the public surface. Cubic interpolation puts 34.4% within four native voxels. Tracking puts 84.4% within the same tolerance. Median surface distance falls from 5.45 to 1.74 voxels. The verifier reads the withheld reference after reconstruction and checks that every original anchor is intact.

## Run on a community mesh

```bash
OMP_NUM_THREADS=4 python fill_sections.py sparse-input.tifxyz /data/native-m7.zarr repaired.tifxyz \
  --axis rows --max-gap 20 --ray-half 80 --fine-step 4 --evidence repair-evidence.npz
```

Supply x.tif, y.tif, z.tif and meta.json from a TIFXYZ mesh. The tool honors mask.tif when present. Missing points can use the conventional −1 coordinates. Each column needs a valid measured point on both sides of a gap. For measured columns with gaps between them, pass `--axis columns`. Known anchors must already have meaningful native coordinates and enough neighboring anchors along the measured section to estimate the local tangent.

Supply an organizer uint8 surface-probability Zarr, either the level-zero array or a group with array `0`. Whole volumes start at world z,y,x = 0,0,0. A cropped array needs the world index of its first voxel through `--origin-zyx`. Set `--max-gap` in mesh-index steps and `--fine-step` in native voxels along the guide. Use a separate output directory.

## Reconstruction

A Hermite curve uses the nearest measured sections on either side of the gap. Local tangents give the native three-dimensional sampling normal. The kernel transports a Gaussian-weighted profile along the curve, once from each end, at two radial window widths. A candidate survives when the two widths agree within four voxels, the endpoint paths differ by at most eight voxels and their matching quality reaches 0.62. The evidence file contains the guide, both agreement diagnostics, the estimated correction and the acceptance mask.

A rejected location stays missing. Gaps with no bracketing points, folds that reverse local orientation and long ambiguous runs require other evidence. Inspect the repaired geometry in cross-section before using it as a reading surface. The unfiltered metrics and the withheld counts below expose the coverage tradeoff.

## Geometry validation

These tests removed all but every eighth or sixteenth section of an already released native mesh. The routine opened only the sparse file and the native M7 Zarr. Scoring measured distance from the recovered coordinate to nearby withheld reference triangles. Each percentage below uses the identical accepted subset for both methods. Multiple views of the same released mesh overlap.

| Released reference | Axis | Anchor interval | Accepted / withheld | Cubic within 4 voxels | Native tracking within 4 voxels |
| --- | --- | ---: | ---: | ---: | ---: |
| PHerc. 1203 / mesh A | rows | 16 | 1,631 / 4,824 | 50.5% | 69.1% |
| PHerc. 1203 / mesh A | rows | 8 | 2,152 / 4,536 | 69.7% | 87.4% |
| PHerc. 1203 / mesh B | rows | 16 | 1,725 / 4,824 | 23.7% | 80.7% |
| PHerc. 1203 / mesh B | rows | 8 | 2,816 / 4,536 | 42.7% | 93.3% |
| PHerc. 1203 / complete mesh A | rows | 16 | 12,333 / 39,671 | 44.1% | 76.3% |
| PHerc. 211 / V6 edge | rows | 16 | 2,878 / 4,018 | 43.3% | 83.1% |
| PHerc. 211 / V6 edge | rows | 8 | 3,154 / 3,732 | 68.8% | 95.2% |
| PHerc. 211 / V6 interior | rows | 16 | 3,718 / 4,288 | 46.5% | 95.3% |
| PHerc. 211 / V6 interior | rows | 8 | 3,501 / 4,032 | 60.9% | 96.4% |
| PHerc. 211 / complete V6 | rows | 16 | 47,495 / 62,708 | 42.3% | 91.0% |
| PHerc. 211 / complete V9 | rows | 16 | 15,224 / 21,431 | 41.5% | 93.4% |
| PHerc. 211 / complete V9 | rows | 8 | 16,667 / 20,034 | 64.4% | 94.4% |
| PHerc. 800 / local field | columns | 16 | 3,284 / 5,071 | 57.2% | 96.3% |
| PHerc. 800 / local field | columns | 8 | 3,981 / 4,733 | 88.0% | 98.7% |
| PHerc. 800 / local field | rows | 16 | 3,540 / 5,039 | 51.4% | 95.7% |
| PHerc. 800 / local field | rows | 8 | 3,912 / 4,734 | 76.3% | 97.1% |
| PHerc. 800 / complete mesh | rows | 16 | 4,905 / 7,645 | 48.4% | 93.8% |

`benchmark/geometry.json` also gives scores on all comparable vertices, the median error and the number of candidates in locations where the reference has no valid geometry. A missing reference cannot confirm or refute a candidate.

## Native CT validation

Each case below used a 240 × 420 chart-pixel field chosen by the size of the predicted correction among supported regions. The hidden geometry and CT scores did not enter window selection. We rendered fifteen actual native CT depths on the released surface, the cubic guide and the repaired surface. The table reports the median of the fifteen matching-plane Pearson correlations. These are local, selected-field measurements.

| Released reference | Comparable image pixels | Cubic CT correlation | Repaired CT correlation |
| --- | ---: | ---: | ---: |
| PHerc. 1203 / mesh A | 18,039 | -0.002 | 0.215 |
| PHerc. 211 / V6 interior | 87,780 | 0.086 | 0.580 |
| PHerc. 800 / local field | 80,680 | 0.390 | 0.838 |

Generate a registered visual comparison on native section 7776 with the following command. It downloads checksum-pinned raw CT directly from the organizer. The four panels show the same measured CT voxels beneath the cubic surface, the reconstructed surface and the withheld reference, with an unmarked copy alongside them.

```bash
python render_example_section.py example-data repaired.tifxyz fit.npz ph211-cross-section.png
```

## Independent comparisons

Four local peak-snapping methods used the same native M7 input and sparse geometry. The following percentages are distances of at most four voxels to the released mesh on identical accepted vertices.

| Public mesh | Accepted vertices | Cubic | Nearest peak | Neighbor-guided peak | Two-sided tracking |
| --- | ---: | ---: | ---: | ---: | ---: |
| PHerc. 211 / complete V9 | 15,224 | 41.5% | 63.8% | 64.2% | 93.4% |
| PHerc. 211 / V6 interior | 3,718 | 46.5% | 68.7% | 69.3% | 95.3% |
| PHerc. 800 / local field | 3,540 | 51.4% | 64.8% | 64.9% | 95.7% |
| PHerc. 1203 / mesh A | 1,631 | 50.5% | 45.1% | 46.0% | 69.1% |
| PHerc. 1203 / mesh B | 1,725 | 23.7% | 27.8% | 28.2% | 80.7% |

The offline example checks the nearest prominent native peak with `python3 compare_local_peaks.py example-data repaired.tifxyz fit.npz`. The published PHerc. 211 winner provides a downstream test on the original 27 native CT planes and the same ink network. At eight-row spacing the correlation with Reader on the original surface rose from 0.935 to 0.956. The correlation with the separately released Studio probabilities fell from 0.697 to 0.680. Exact measurements are in `benchmark/known-211-downstream.json`.

## Reproduce the complete PHerc. 211 benchmark

This benchmark used the 87 × 301 normalized V9 geometry in Priyank Aranke’s released PHerc. 211 submission. The [winner’s V9 mesh archive](https://drive.google.com/file/d/1aQhMxc3A5wYOBtHGy9fiJaGUyNwDzLmu/view) contains the four files in `mesh/normalized-tifxyz/`. The separate catalog export is 86 × 300 and produces different withholding counts.

```bash
curl --fail --location 'https://drive.usercontent.google.com/download?id=1aQhMxc3A5wYOBtHGy9fiJaGUyNwDzLmu&export=download&confirm=t' -o winner-v9-mesh.zip
echo '819dcfb40451fab3fa58441d8265813c8ddbe2b5693bb076fce3eeefce05a5d9  winner-v9-mesh.zip' | sha256sum --check
unzip -j winner-v9-mesh.zip 'mesh/normalized-tifxyz/*.tif' 'mesh/normalized-tifxyz/meta.json' -d ph211-v9.tifxyz
python fetch_benchmark_inputs.py ph211-native-m7.zarr
OMP_NUM_THREADS=4 python benchmark_mesh.py ph211-v9.tifxyz ph211-native-m7.zarr ph211-gap16 --gap 16 --local-peaks
```

The benchmark starts with every sixteenth row, scores matching 3D triangles around each withheld vertex, and writes the sparse inputs, recovered mesh, acceptance mask and scores. The published geometry table includes accepted and unfiltered counts. For a smaller offline exercise, pass `example-data/published-reference.tifxyz example-data/native_m7.zarr offline-benchmark --gap 8 --origin-zyx 7444 3315 3198 --local-peaks` to the same script.

At an eight-row interval, the repair increased agreement with Reader run on the original geometry from 0.935 to 0.956. Agreement with the independently released Studio map fell from 0.697 to 0.680 on the same pixels. Better geometry alone does not establish better legibility.

## Sources and attribution

The example uses Vesuvius Challenge native M7 on PHerc. 211 and a cropped parent surface from Priyank Aranke's public First Letters materials. The preparation and rendering scripts obtain the scan inputs from their original publishers. The CT, M7 excerpt, source meshes and locally generated illustrations retain their source terms. The native collection is distributed under CC BY-NC 4.0.

- [Official data and attribution](https://scrollprize.org/data)
- [Native PHerc. 211 scan and predictions](https://scrollprize.org/data_browser/PHerc0211)
- [Priyank Aranke's released PHerc. 211 materials](https://drive.google.com/drive/folders/1bvKolL_TIE3k5Ixi1qlOiAmofQ6r2ocH)
- [Native PHerc. 800 surfaces](https://scrollprize.org/data_browser/PHerc0800)
- [Native PHerc. 1203 data](https://scrollprize.org/data_browser/PHerc1203)
- [CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/)

Angelotti et al., *Vesuvius Challenge — CT Scans of Herculaneum Papyri*, Vesuvius Challenge. The native crop coordinates are in `example/volume.json`. The M7 and raw CT source hashes are in `example/pherc211-native-inputs.json` and `example/pherc211-ct-section-inputs.json`.

Copyright © 2026 Dennis Koshta. The original code is available under the [MIT License](LICENSE) and was developed with AI assistance. The scan excerpts, geometry, and derived illustrations are covered by their source licenses as described above.
