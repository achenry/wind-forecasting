"""Tests for the fixes on branch fix/critical-forecasting-bugs.

1. Future dynamic-real features (nacelle direction nd_cos/nd_sin) are persisted from the last
   observed value instead of leaking the realised future (train + val datasets).
2. The training IterableDataset is shuffled, re-shuffled every epoch, covers every window
   exactly once per epoch, and stays consistent across DataLoader workers.
3. With repeat=True, feat_static_cat stays aligned with its series across passes.
4. Tuning and training read the training-sampler type from the same place.

Run with:  python -m pytest tests/test_critical_fixes.py -q
"""
import numpy as np
import polars as pl
import pytest
import torch
from gluonts.time_feature._base import minute_of_hour, second_of_minute

from wind_forecasting.preprocessing.pytorch_dataset import (
    WindForecastingDatamodule,
    WindForecastingDataset,
)
from wind_forecasting.tuning.utils.helpers import get_train_sampler_type

CTX, PRED = 8, 4
LENGTHS = [40, 30, 25]  # one continuity group / item each
TIME_FEATURES = [second_of_minute, minute_of_hour]
N_TF = len(TIME_FEATURES)


def _make_split(path):
    frames = []
    for item, n in enumerate(LENGTHS):
        t = np.arange(n)
        frames.append(
            pl.DataFrame(
                {
                    "item_id": [f"ITEM{item}"] * n,
                    "time": pl.datetime_range(
                        pl.datetime(2024, 1, 1 + item), pl.datetime(2024, 1, 1 + item)
                        + pl.duration(seconds=15 * (n - 1)), "15s", eager=True),
                    # encode (item, t) in the target so samples are identifiable
                    "target_0": (item * 1000 + t).astype(np.float32),
                    "target_1": (item * 1000 + t + 0.5).astype(np.float32),
                    # dynamic reals vary in time so leakage is detectable
                    "feat_dynamic_real_0": (100 + t).astype(np.float32),
                    "feat_dynamic_real_1": (-100 - t).astype(np.float32),
                    "feat_static_cat": [[item]] * n,
                }
            )
        )
    pl.concat(frames).write_parquet(path)


@pytest.fixture()
def datamodule(tmp_path):
    train_p, val_p = tmp_path / "train.parquet", tmp_path / "val.parquet"
    _make_split(train_p)
    _make_split(val_p)
    dm = WindForecastingDatamodule(
        train_data_path=str(train_p), val_data_path=str(val_p), train_sampler=None,
        context_length=CTX, prediction_length=PRED, time_features=TIME_FEATURES,
        batch_size=4, num_workers=0, pin_memory=False, shuffle_buffer_length=16,
    )
    dm.setup("fit")
    return dm


def _sample_key(sample):
    past_target, feat_static_cat = sample[0], sample[6]
    return int(feat_static_cat.item()), int(past_target[0, 0].item()) % 1000


def _all_windows():
    return {(item, s) for item, n in enumerate(LENGTHS) for s in range(0, n - CTX - PRED + 1)}


def _train_dataset(dm, **kw):
    kwargs = dict(data=dm.train_data, context_length=CTX, prediction_length=PRED, sampler=None,
                  shuffle=True, shuffle_buffer_length=16, n_time_features=N_TF)
    kwargs.update(kw)
    return WindForecastingDataset(**kwargs)


# ---------------------------------------------------------------- 1. leakage
@pytest.mark.parametrize("split", ["train", "val"])
def test_future_dynamic_reals_are_persisted(datamodule, split):
    loader = datamodule.train_dataloader() if split == "train" else datamodule.val_dataloader()
    n = 0
    for batch in loader:
        past_tf, future_tf = batch[2], batch[3]
        # dynamic-real columns: every future step == last observed value
        expected = past_tf[:, -1:, N_TF:].expand(-1, PRED, -1)
        assert torch.equal(future_tf[:, :, N_TF:], expected)
        # calendar time features are NOT persisted: they must keep advancing
        assert not torch.equal(future_tf[:, :, :N_TF], past_tf[:, -1:, :N_TF].expand(-1, PRED, -1))
        n += past_tf.shape[0]
    assert n > 0


def test_persistence_can_be_disabled(datamodule):
    ds = _train_dataset(datamodule, shuffle=False, persist_future_feat_dynamic_real=False)
    sample = next(iter(ds))
    past_tf, future_tf = sample[2], sample[3]
    # original (leaky) behaviour: future dynamic reals continue the ramp
    assert future_tf[0, N_TF] == past_tf[-1, N_TF] + 1


# ---------------------------------------------------------------- 2. shuffling
def test_each_epoch_covers_all_windows_once_and_reshuffles(datamodule):
    loader = datamodule.train_dataloader()
    epochs = []
    for _ in range(3):
        keys = [(int(c), int(t) % 1000) for b in loader
                for c, t in zip(b[6].flatten(), b[0][:, 0, 0])]
        assert len(keys) == len(set(keys)), "duplicate windows within one epoch"
        assert set(keys) == _all_windows(), "epoch does not cover every window"
        epochs.append(keys)
    assert epochs[0] != sorted(epochs[0]), "epoch is not shuffled"
    assert epochs[0] != epochs[1] and epochs[1] != epochs[2], "same order every epoch"


def test_workers_share_one_permutation(datamodule):
    """Each worker islices the same stream: union must be exactly one copy of every window."""
    datamodule.num_workers = 2
    loader = datamodule.train_dataloader()
    for _ in range(2):
        keys = [(int(c), int(t) % 1000) for b in loader
                for c, t in zip(b[6].flatten(), b[0][:, 0, 0])]
        assert sorted(keys) == sorted(_all_windows())


def test_unshuffled_order_unchanged(datamodule):
    ds = _train_dataset(datamodule, shuffle=False)
    keys = [_sample_key(s) for s in ds]
    assert keys == sorted(_all_windows())


# ---------------------------------------------------------------- 3. repeat alignment
def test_repeat_keeps_static_cat_aligned(datamodule):
    ds = _train_dataset(datamodule, repeat=True)
    it = iter(ds)
    n_per_pass = len(_all_windows())
    for _ in range(4 * n_per_pass):
        sample = next(it)
        item_from_target = int(sample[0][0, 0].item()) // 1000
        assert int(sample[6].item()) == item_from_target


# ---------------------------------------------------------------- 4. sampler config
@pytest.mark.parametrize(
    "config, expected",
    [
        ({"dataset": {}, "optuna": {}}, "random"),
        ({"dataset": {}, "optuna": {"sampler": "sequential"}}, "sequential"),
        ({"dataset": {"sampler": "sequential"}, "optuna": {"sampler": "random"}}, "sequential"),
        ({"dataset": {"sampler": "random"}, "optuna": {"sampler": "sequential"}}, "random"),
    ],
)
def test_train_sampler_type_single_source(config, expected):
    assert get_train_sampler_type(config) == expected


def test_train_sampler_type_rejects_unknown():
    with pytest.raises(ValueError):
        get_train_sampler_type({"dataset": {"sampler": "tpe"}, "optuna": {}})
