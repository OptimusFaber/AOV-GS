# `third_parties/`

Python dependencies for AOV-GS. **Not stored in git** (same as [ActiveSGM](https://github.com/lly00412/ActiveSGM)) — fetched after clone.

## Setup

```bash
git clone --recursive https://github.com/OptimusFaber/AOV-GS
cd AOV-GS
```

If you cloned without `--recursive`:

```bash
bash scripts/installation/setup_third_parties.sh
```

Or manually: `git submodule update --init --recursive` (see [`.gitmodules`](../.gitmodules)).

## Directories

| Path | Purpose | Needed for |
|------|------------|-----------|
| `coslam/` | Co-SLAM utils | visualization, RRT |
| `splatam/` | SplaTAM helpers | SLAM, NVS |
| `neural_slam_eval/` | mesh / recon metrics | eval_recon |
| `channel_rasterization/` | dense semantic rasterizer | **ActiveSem** (optional) |
| `sparse_channel_rasterization/` | sparse semantic rasterizer | **ActiveSem** (optional) |

For **`ActiveOpenSem`** + language field, `coslam`, `splatam`, and `neural_slam_eval` are enough (~150 MB).

## `habitat_sim/`

Not part of the `third_parties/` git tree. Habitat is installed via conda: `bash scripts/installation/conda_env/build_sem.sh`.
