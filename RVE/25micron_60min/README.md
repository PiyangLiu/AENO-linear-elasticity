# Bristol `25micron_60min` XCT volume

This directory is a local working location for the third-party XCT slices used
to construct the CT-RVE benchmark in the AENO manuscript. The raw TIF slices
are not redistributed as part of this code release. For public archiving, keep
this README and obtain the raw data from the original public repository.

## Source dataset

The CT-RVE benchmark uses the Bristol `25micron_60min` XCT volume of a
void-containing unidirectional carbon-fibre/epoxy laminate. The volume is part
of the public University of Bristol dataset:

```text
X-Ray Computed Tomography Scans of a Composite Laminate with Voids
University of Bristol
DOI: https://doi.org/10.5523/bris.1ywqtm9gy6fgl2lwz17ukn2obr
```

Please cite the original Bristol dataset when using or reconstructing the
CT-RVE benchmark. The raw XCT data remain subject to the reuse terms of the
original dataset record and are not relicensed by this repository.

## What was used in this study

For the CT-RVE benchmark, TIF slices from the `25micron_60min` XCT volume were
stacked into a three-dimensional greyscale volume of approximately
`240 x 240 x 240` voxels, with axis order `(z, y, x)`.

Non-overlapping `32^3` representative-volume-element (RVE) crops were generated
with stride 32. The normalized XCT greyscale volume was segmented using a global
low-intensity quantile threshold:

```text
threshold quantile: 2%
threshold value:    T = 0.5455
void phase:         chi_v = 1 where I(x) <= T
solid phase:        chi_v = 0 where I(x) > T
```

Crops with porosity below `1e-3` were excluded. The retained processed dataset
contains 325 RVEs, with porosities ranging from `0.116%` to `10.50%` and a mean
porosity of `2.34%`.

## Spatial split

The retained RVEs were split by non-overlapping crop-start layers along the
`z` direction:

```text
training:   z = 0, 32, 64, 96, 128   237 samples
validation: z = 160                   47 samples
test:       z = 192                   41 samples
```

The test set is therefore a same-volume spatial holdout from the same XCT scan.
It should not be interpreted as transfer to independent tomographic specimens,
different imaging protocols or different material systems.

## Reproducing the processed benchmark file

From the `RVE` directory, the processed CT-RVE dataset can be reconstructed
from the locally downloaded Bristol TIF slices with:

```bash
python make_dataset.py \
  --tif_dir 25micron_60min \
  --out data/bristol_25um_rve32_spatial_gray.npz \
  --crop_size 32 \
  --stride 32 \
  --target_porosity 0.02 \
  --min_porosity 0.001 \
  --max_rves 0 \
  --split_mode spatial_z_blocks
```

The resulting processed files used by the manuscript workflow are stored under
`RVE/data/`:

```text
bristol_25um_rve32_spatial_gray.npz
bristol_25um_rve32_spatial_gray.json
bristol_25um_rve32_spatial_gray.index.csv
```

## Public-release note

Before depositing this repository in a public archive, remove the raw
third-party files from this directory, including:

```text
slice_*.tif
.header.html
.footer.html
```

Keep this README so that users can retrieve the source data from the original
Bristol DOI and reconstruct the processed CT-RVE benchmark with the documented
pipeline.
