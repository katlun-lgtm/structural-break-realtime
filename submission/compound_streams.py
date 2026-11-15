"""Causal streaming ports for the SBRT v7 stage-expert additions.

The deployed tree already receives Build K's four TrimGap columns.  This module
adds the remaining sixteen columns used by the early 141-column v5 expert:
MAC(4), MBD(3), TP(2), IDR(3), then dynamic bin mode(4).  TP and IDR are
deliberately zero for online indices 0..63.  The late v4 expert consumes only
the first twelve columns and ignores bin mode.
"""

from __future__ import annotations

import heapq
from typing import List, Sequence

import numpy as np

from mac_xrate_stream import MAC_XRATE_NAMES, MACXRateStream, batch_mac_xrate


MBD_NAMES = ["mbd_run_ratio_k10", "mbd_current_run_k10", "mbd_entry_ratio_k10"]
TP_NAMES = ["tp_velocity_density_split", "tp_acceleration_density_split"]
IDR_NAMES = ["idr_online_ratio_p2", "idr_contrast_p2", "idr_log_ratio_p2"]
BIN_MODE_NAMES = [
    "bin_mode_mean_run_dyn",
    "bin_mode_current_run_dyn",
    "bin_mode_mean_contrast_dyn",
    "bin_mode_mean_ratio_dyn",
]
COMPOUND_EXTRA_NAMES = (
    list(MAC_XRATE_NAMES) + MBD_NAMES + TP_NAMES + IDR_NAMES + BIN_MODE_NAMES
)
TP_START = 64
IDR_START = 64


def _push_median(low: List[float], high: List[float], value: float) -> float:
    if not low or value <= -low[0]:
        heapq.heappush(low, -value)
    else:
        heapq.heappush(high, value)
    if len(low) > len(high) + 1:
        heapq.heappush(high, -heapq.heappop(low))
    elif len(high) > len(low):
        heapq.heappush(low, -heapq.heappop(high))
    if len(low) > len(high):
        return float(-low[0])
    return float((-low[0] + high[0]) / 2.0)


class MBDDwellStream:
    """Streaming port of ``gen_mbd_dwell_contrast_k10``."""

    __slots__ = (
        "low",
        "high",
        "band",
        "hist_mean_run",
        "hist_entry_rate",
        "cur_run",
        "completed_sum",
        "completed_count",
        "entries",
        "t",
    )

    def __init__(self, x_historical: Sequence[float]):
        history = np.asarray(x_historical, dtype=np.float64).ravel()
        hist_med = float(np.median(history)) if history.size else 0.0
        mad = (
            float(np.median(np.abs(history - hist_med))) * 1.4826
            if history.size
            else 1.0
        )
        self.band = max(mad, 1e-6)
        self.low: List[float] = []
        self.high: List[float] = []
        hist_in = []
        for value in history:
            median = _push_median(self.low, self.high, float(value))
            hist_in.append(abs(float(value) - median) <= self.band)

        runs = []
        current = 0
        for inside in hist_in:
            if inside:
                current += 1
            else:
                if current:
                    runs.append(current)
                current = 0
        if current:
            runs.append(current)
        self.hist_mean_run = max(float(np.mean(runs)) if runs else 1.0, 1e-3)
        self.hist_entry_rate = len(runs) / max(len(history), 1)
        self.cur_run = current
        self.completed_sum = 0.0
        self.completed_count = 0
        self.entries = 0
        self.t = 0

    def step(self, value: float) -> List[float]:
        median = _push_median(self.low, self.high, float(value))
        inside = abs(float(value) - median) <= self.band
        if inside:
            self.cur_run += 1
        else:
            if self.cur_run:
                self.completed_sum += self.cur_run
                self.completed_count += 1
                self.entries += 1
            self.cur_run = 0

        if self.completed_count:
            online_mean_run = self.completed_sum / self.completed_count
        else:
            online_mean_run = float(self.cur_run) if self.cur_run else 1.0
        online_mean_run = max(online_mean_run, 1e-3)
        ratio = float(np.clip(online_mean_run / self.hist_mean_run, 0.0, 20.0))
        cur_norm = float(np.clip(self.cur_run / self.hist_mean_run, 0.0, 20.0))
        entry_rate = self.entries / float(self.t + 1)
        entry_ratio = float(
            np.clip(entry_rate / max(self.hist_entry_rate, 1e-6), 0.0, 20.0)
        )
        row = [ratio, cur_norm, entry_ratio]
        if self.t < 5:
            row = [0.0, 0.0, 0.0]
        self.t += 1
        return np.asarray(row, dtype=np.float32).tolist()


def batch_mbd_dwell_contrast_k10(
    x_historical: Sequence[float], x_online: Sequence[float]
) -> np.ndarray:
    stream = MBDDwellStream(x_historical)
    rows = [stream.step(float(value)) for value in x_online]
    if not rows:
        return np.zeros((0, len(MBD_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class TPOrderDensityStream:
    """Streaming port of ``gen_tp_order_density_split`` with the v3 gate."""

    __slots__ = (
        "dens_vel_h",
        "dens_acc_h",
        "last_x",
        "prev_dx",
        "prev_v_sign",
        "prev_a_sign",
        "vel_count",
        "acc_count",
        "t",
    )

    def __init__(self, x_historical: Sequence[float]):
        history = np.asarray(x_historical, dtype=np.float64).ravel()
        if history.size >= 3:
            dh = np.diff(history)
            signs = np.sign(dh)
            nonzero = signs != 0
            vel_h = float(
                ((signs[1:] != signs[:-1]) & nonzero[1:] & nonzero[:-1]).sum()
            )
            d2h = np.diff(dh)
            signs2 = np.sign(d2h)
            nonzero2 = signs2 != 0
            acc_h = float(
                ((signs2[1:] != signs2[:-1]) & nonzero2[1:] & nonzero2[:-1]).sum()
            )
        else:
            vel_h = 0.0
            acc_h = 0.0
        self.dens_vel_h = vel_h / max(history.size - 1, 1)
        self.dens_acc_h = acc_h / max(history.size - 2, 1)

        if history.size >= 2:
            first, second = float(history[-2]), float(history[-1])
        elif history.size == 1:
            first = second = float(history[0])
        else:
            first = second = 0.0
        self.last_x = second
        self.prev_dx = second - first
        self.prev_v_sign = float(np.sign(self.prev_dx))
        self.prev_a_sign = None
        self.vel_count = 0.0
        self.acc_count = 0.0
        self.t = 0

    def step(self, value: float) -> List[float]:
        x = float(value)
        dx = x - self.last_x
        v_sign = float(np.sign(dx))
        if v_sign != self.prev_v_sign and v_sign != 0.0 and self.prev_v_sign != 0.0:
            self.vel_count += 1.0

        a_sign = float(np.sign(dx - self.prev_dx))
        if (
            self.prev_a_sign is not None
            and a_sign != self.prev_a_sign
            and a_sign != 0.0
            and self.prev_a_sign != 0.0
        ):
            self.acc_count += 1.0

        count = self.t + 1
        dens_vel_o = self.vel_count / float(count)
        dens_acc_o = self.acc_count / float(max(self.t, 1))
        eps = 1e-6
        col_a = float(np.log((dens_vel_o + eps) / (self.dens_vel_h + eps)))
        col_b = float(np.log((dens_acc_o + eps) / (self.dens_acc_h + eps)))
        row = np.nan_to_num(
            np.clip(np.asarray([col_a, col_b]), -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).astype(np.float32)
        if self.t < 3 or self.t < TP_START:
            row[:] = 0.0

        self.last_x = x
        self.prev_dx = dx
        self.prev_v_sign = v_sign
        self.prev_a_sign = a_sign
        self.t += 1
        return row.tolist()


def batch_tp_order_density_split(
    x_historical: Sequence[float], x_online: Sequence[float]
) -> np.ndarray:
    stream = TPOrderDensityStream(x_historical)
    rows = [stream.step(float(value)) for value in x_online]
    if not rows:
        return np.zeros((0, len(TP_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class _P2State:
    """The frozen P² marker update used by ``idr_std_ratio_p2``."""

    __slots__ = ("p", "n", "desired", "delta", "heights", "buffer")

    def __init__(self, probability: float):
        p = float(probability)
        self.p = p
        self.n = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        self.desired = np.asarray([1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0])
        self.delta = np.asarray([0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0])
        self.heights = None
        self.buffer = []

    def update(self, value: float) -> None:
        x = float(value)
        if self.heights is None:
            self.buffer.append(x)
            if len(self.buffer) == 5:
                self.heights = np.sort(np.asarray(self.buffer, dtype=np.float64))
            return
        h, n, desired, delta = self.heights, self.n, self.desired, self.delta
        if x < h[0]:
            h[0] = x
            k = 0
        elif x >= h[4]:
            h[4] = x
            k = 3
        else:
            k = 0
            for index in range(4):
                if h[index] <= x < h[index + 1]:
                    k = index
                    break
        for index in range(k + 1, 5):
            n[index] += 1.0
        for index in range(5):
            desired[index] += delta[index]
        for index in range(1, 4):
            distance = desired[index] - n[index]
            if (
                distance >= 1.0
                and n[index + 1] - n[index] > 1.0
                or distance <= -1.0
                and n[index - 1] - n[index] < -1.0
            ):
                direction = 1.0 if distance >= 1.0 else -1.0
                denominator = n[index + 1] - n[index - 1]
                term = direction * (
                    (n[index] - n[index - 1] + direction)
                    * (h[index + 1] - h[index])
                    / (n[index + 1] - n[index])
                    + (n[index + 1] - n[index] - direction)
                    * (h[index] - h[index - 1])
                    / (n[index] - n[index - 1])
                )
                proposed = h[index] + term / denominator
                if h[index - 1] < proposed < h[index + 1]:
                    h[index] = proposed
                else:
                    neighbor = index + int(direction)
                    h[index] = h[index] + direction * (h[neighbor] - h[index]) / (
                        n[neighbor] - n[index]
                    )
                n[index] += direction

    def quantile(self) -> float | None:
        if self.heights is not None:
            return float(self.heights[2])
        if not self.buffer:
            return None
        values = np.sort(np.asarray(self.buffer, dtype=np.float64))
        position = self.p * (len(values) - 1)
        lower, upper = int(np.floor(position)), int(np.ceil(position))
        if lower == upper:
            return float(values[lower])
        fraction = position - lower
        return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


class IDRStdRatioStream:
    """Streaming port of ``gen_idr_std_ratio_p2``, gated through index 63."""

    __slots__ = ("hist_ratio", "low", "high", "mean", "m2", "seen")

    def __init__(self, x_historical: Sequence[float]):
        history = np.nan_to_num(np.asarray(x_historical, dtype=np.float64).ravel())
        eps = 1e-8
        if history.size >= 5:
            q10 = float(np.percentile(history, 10))
            q90 = float(np.percentile(history, 90))
            std = float(np.std(history))
            self.hist_ratio = float(np.clip((q90 - q10) / (std + eps), 0.0, 10.0))
        else:
            self.hist_ratio = 2.56
        self.low = _P2State(0.10)
        self.high = _P2State(0.90)
        self.mean = 0.0
        self.m2 = 0.0
        self.seen = 0

    def step(self, value: float) -> List[float]:
        x = float(np.nan_to_num(np.float64(value)))
        self.low.update(x)
        self.high.update(x)
        self.seen += 1
        delta = x - self.mean
        self.mean += delta / self.seen
        self.m2 += delta * (x - self.mean)
        row = np.zeros(3, dtype=np.float64)
        if self.seen >= 8:
            std = float(np.sqrt(max(self.m2 / self.seen, 0.0)))
            q10 = self.low.quantile()
            q90 = self.high.quantile()
            if q10 is not None and q90 is not None:
                eps = 1e-8
                online_ratio = float(np.clip((q90 - q10) / (std + eps), 0.0, 10.0))
                row[0] = online_ratio
                row[1] = float(np.clip(online_ratio - self.hist_ratio, -10.0, 10.0))
                row[2] = float(
                    np.clip(
                        np.log((online_ratio + eps) / (self.hist_ratio + eps)),
                        -5.0,
                        5.0,
                    )
                )
        row = np.clip(
            np.nan_to_num(row, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0
        ).astype(np.float32)
        if self.seen <= IDR_START:
            row[:] = 0.0
        return row.tolist()


def batch_idr_std_ratio_p2(
    x_historical: Sequence[float], x_online: Sequence[float]
) -> np.ndarray:
    stream = IDRStdRatioStream(x_historical)
    rows = [stream.step(float(value)) for value in x_online]
    if not rows:
        return np.zeros((0, len(IDR_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class BinModeConvStreakStream:
    """Streaming port of the ungated dynamic-bin convergence streak."""

    __slots__ = (
        "active",
        "edges",
        "nb",
        "counts",
        "running_mode",
        "prev_dist",
        "run_len",
        "run_sum",
        "run_count",
        "hist_baseline",
        "t",
    )

    def __init__(self, x_historical: Sequence[float]):
        history = np.asarray(x_historical, dtype=np.float64).ravel()
        self.active = history.size >= 5
        self.edges = np.asarray([], dtype=np.float64)
        self.nb = 0
        self.counts = np.asarray([], dtype=np.int64)
        self.running_mode = 0
        self.prev_dist = 0
        self.run_len = 0
        self.run_sum = 0.0
        self.run_count = 0
        self.hist_baseline = 0.0
        self.t = 0
        if not self.active:
            return
        quantiles = np.linspace(0.0, 1.0, 11)[1:-1]
        self.edges = np.unique(np.quantile(history, quantiles))
        self.nb = len(self.edges) + 1
        if self.nb < 2:
            self.active = False
            return

        def bin_of(value: float) -> int:
            return int(np.clip(np.searchsorted(self.edges, value, side="right"), 0, self.nb - 1))

        self.counts = np.zeros(self.nb, dtype=np.int64)
        for value in history:
            self.counts[bin_of(float(value))] += 1
        self.running_mode = int(np.argmax(self.counts))

        hist_counts = np.zeros(self.nb, dtype=np.int64)
        previous_bin = bin_of(float(history[0]))
        hist_counts[previous_bin] += 1
        hist_mode = int(np.argmax(hist_counts))
        previous_dist = abs(previous_bin - hist_mode)
        run_len = 0
        run_sum = 0.0
        run_count = 0
        for value in history[1:]:
            current_bin = bin_of(float(value))
            distance = abs(current_bin - hist_mode)
            if distance < previous_dist:
                run_len += 1
            else:
                if run_len:
                    run_sum += run_len
                    run_count += 1
                run_len = 0
            previous_dist = distance
            hist_counts[current_bin] += 1
            hist_mode = int(np.argmax(hist_counts))
        if run_len:
            run_sum += run_len
            run_count += 1
        self.hist_baseline = run_sum / run_count if run_count else 0.0
        last_bin = bin_of(float(history[-1]))
        self.prev_dist = abs(last_bin - self.running_mode)

    def step(self, value: float) -> List[float]:
        if not self.active:
            self.t += 1
            return [0.0, 0.0, 0.0, 0.0]
        current_bin = int(
            np.clip(
                np.searchsorted(self.edges, float(value), side="right"),
                0,
                self.nb - 1,
            )
        )
        distance = abs(current_bin - self.running_mode)
        if distance < self.prev_dist:
            self.run_len += 1
        else:
            if self.run_len:
                self.run_sum += self.run_len
                self.run_count += 1
            self.run_len = 0
        self.prev_dist = distance
        self.counts[current_bin] += 1
        self.running_mode = int(np.argmax(self.counts))

        denominator = self.run_count + (1 if self.run_len > 0 else 0)
        mean_run = (
            (self.run_sum + self.run_len) / denominator if denominator > 0 else 0.0
        )
        row = np.asarray(
            [
                np.clip(mean_run, 0.0, 20.0),
                np.clip(float(self.run_len), 0.0, 20.0),
                np.clip(mean_run - self.hist_baseline, -20.0, 20.0),
                np.clip(mean_run / (self.hist_baseline + 1e-6), 0.0, 20.0),
            ],
            dtype=np.float32,
        )
        self.t += 1
        return row.tolist()


def batch_bin_mode_conv_streak(
    x_historical: Sequence[float], x_online: Sequence[float]
) -> np.ndarray:
    stream = BinModeConvStreakStream(x_historical)
    rows = [stream.step(float(value)) for value in x_online]
    if not rows:
        return np.zeros((0, len(BIN_MODE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def batch_compound_extra(
    x_historical: Sequence[float], x_online: Sequence[float]
) -> np.ndarray:
    blocks = [
        batch_mac_xrate(x_historical, x_online),
        batch_mbd_dwell_contrast_k10(x_historical, x_online),
        batch_tp_order_density_split(x_historical, x_online),
        batch_idr_std_ratio_p2(x_historical, x_online),
        batch_bin_mode_conv_streak(x_historical, x_online),
    ]
    return np.hstack(blocks).astype(np.float32, copy=False)
