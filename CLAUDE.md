# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official implementation of "Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises" (IEEE TNSRE, 2022). It predicts a continuous quality/assessment score for a physical rehabilitation exercise from 3D skeleton (Kinect) motion-capture sequences, using an extended ST-GCN + LSTM model. This is research code, not a package/library — no build system, no tests, no CI.

## Setup & commands

Use Python 3.7.17 via pyenv, in the `stgcn-fk-old-env` pyenv-virtualenv (this matches the pinned old TensorFlow/Keras versions in `requirements.txt`):

```bash
pyenv activate stgcn-fk-old-env   # or: pyenv shell stgcn-fk-old-env
python --version                  # should report 3.7.17
```

```bash
pip install -r requirements.txt   # also requires TensorFlow 2.x installed separately per README
python demo.py                    # inference demo using pretrained weights + Data/input.csv, Data/label.csv
python demo.py [--inputs PATH] [--labels PATH]   # demo.py uses tf.app.flags, NOT argparse; flags are --inputs/--labels
python train.py --ex Kimore_ex5 --epoch 2000 --batch_size 10   # --ex is required; --lr and --epoch also available
```

There is no lint config, no test suite, and no CI in this repo. Do not invent test/lint commands.

`train.py --ex <name>` expects a folder named `<name>/` (relative to cwd) containing `Train_X.csv` and `Train_Y.csv` (see `GCN/data_processing.py: import_dataset`). These per-exercise dataset folders (from KIMORE / UI-PRMD, linked in README.md) are not checked into the repo — only the small demo files under `Data/` and the pretrained model under `pretrain model/` are.

## Architecture

Data flow: raw Kinect CSV (97 channels: 25 joints × orientation+position) → joint selection/reorder → per-feature `StandardScaler` → reshape to `(batch, 100 timesteps, 25 joints, 3 channels)` → ST-GCN+LSTM model → single linear score, inverse-scaled back to original units.

- **`GCN/graph.py`** (`Graph`): builds the fixed skeleton adjacency for 25 joints (hardcoded Kinect topology as 1-based joint pairs, converted to 0-based). Produces `AD` (1-hop adjacency), `AD2` (2-hop adjacency), and `bias_mat_1`/`bias_mat_2` — additive attention masks that are `0` where an edge exists and `-1e9` elsewhere, used to zero out non-neighbor attention weights after softmax.
- **`GCN/data_processing.py`** (`Data_Loader`): loads `Train_X.csv`/`Train_Y.csv` for a given exercise, reorders raw 97-channel Kinect data down to a specific 25-joint × 3-channel subset (joint index list is duplicated — see below), fits and applies `StandardScaler` to both X and Y, reshapes into `(batch, 100, 25, 3)`. `num_timestep` is hardcoded to 100 — dataset preprocessing upstream must produce sequences of exactly that length.
- **`GCN/sgcn_lstm.py`** (`Sgcn_Lstm`): the model.
  - `sgcn(Input)`: one ST-GCN block — temporal `Conv2D` → concat with input → two parallel graph-attention "hops" (1-hop using `bias_mat_1`/`AD`-derived masking, 2-hop using `bias_mat_2`) where attention logits come from a `ConvLSTM2D` and are combined via masked softmax + `einsum` aggregation over joints → concat the two hops → multi-scale temporal convs (kernel sizes 9/15/20) concatenated together.
  - `train()`: stacks three `sgcn` blocks with residual connections (`y = sgcn(x) + x`, `z = sgcn(y) + y`), feeds the result through `Lstm()` (4 stacked LSTM layers, dropout between each, final `Dense(1, linear)`), builds the Keras `Model`, and fits it with `Huber` loss + `ModelCheckpoint`.
  - `prediction(data)`: only valid after `train()` has been called on the same instance (it uses `self.model` set there) — there is no way to load an `Sgcn_Lstm` from saved weights directly; that's why `demo.py` reconstructs the model independently instead of reusing this class.
- **`train.py`**: CLI entrypoint (argparse) wiring `Data_Loader` → `Graph` → 80/20 `train_test_split` → `Sgcn_Lstm.train()` → reports MAE/RMSE/MSE/MAPE on inverse-transformed predictions.
- **`demo.py`**: standalone inference path. Uses `tf.compat.v1.app.flags` (not argparse) for `--inputs`/`--labels`. Re-implements the same 25-joint reorder logic from `data_processing.py` (kept in sync manually — there is no shared module for this), loads the saved `Data/sc_x.save` / `Data/sc_y.save` joblib scalers (fit during original training, not refit here), reconstructs the model from `pretrain model/rehabilitation.json` + `pretrain model/best_model.hdf5`, and prints predicted vs. actual score for the first sample.

## Known sharp edges (be aware, don't "fix" silently)

- The 25-item Kinect joint-index list and the reorder loop are duplicated verbatim in `demo.py` and `GCN/data_processing.py`. If you change one, change the other.
- `Sgcn_Lstm.train()` hardcodes the checkpoint path `"best model ex4/best_model.hdf5"` regardless of `--ex` — training a different exercise will still write there (and the directory must already exist).
- `StandardScaler` in `Data_Loader.preprocessing()` is fit on the full dataset before `train.py` does its train/test split — be aware of this when reasoning about metrics.
- Pinned to old TensorFlow/Keras (`tensorflow==2.4.0`, `keras==2.4.3`); APIs like `Adam(lr=...)` and `ModelCheckpoint(..., period=1)` are deprecated/removed in newer TF/Keras — don't "modernize" these incidentally unless asked, since the pinned versions are what the pretrained weights/model JSON were built against.
