"""
拍点 / 小节正拍提取

优先：madmom_infer = CPJKU/madmom 推理重实现
  RNNDownBeatProcessor + DBNDownBeatTrackingProcessor
  （Böck et al., ISMIR 2016 joint beat/downbeat；MIREX 常用 DBN 后处理）

回退：librosa Ellis DP beat_track + onset 邻域校正 + 正拍相位选择
  （无神经网络时）

参考：
- https://github.com/CPJKU/madmom
- https://github.com/openmirlab/madmom-infer
- ISMIR 2024 Beat This!（更强但需 torch；此处用可 CPU 跑的 madmom-infer）
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _librosa_beats(y: np.ndarray, sr: int, hop_length: int = 512) -> tuple[float, np.ndarray, np.ndarray]:
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    tempo_curve = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None
    )
    tempo_arr, beat_times = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        bpm=tempo_curve,
        tightness=100,
        trim=True,
        units="time",
    )
    beat_times = np.asarray(beat_times, dtype=float)
    tempo = float(np.median(np.atleast_1d(tempo_arr).astype(float)))
    if len(beat_times) < 2:
        return tempo, beat_times, beat_times[:1].copy()

    # 邻域吸附到 onset 峰（madmom correct=True 同类）
    duration = float(librosa.get_duration(y=y, sr=sr))
    med_ibi = float(np.median(np.diff(beat_times)))
    win = float(np.clip(med_ibi * 0.12, 0.04, 0.10))
    snapped = []
    for t in beat_times:
        f0 = int(librosa.time_to_frames(max(0.0, t - win), sr=sr, hop_length=hop_length))
        f1 = int(librosa.time_to_frames(min(duration, t + win), sr=sr, hop_length=hop_length))
        f0 = max(0, f0)
        f1 = min(len(onset_env), max(f0 + 1, f1))
        peak = f0 + int(np.argmax(onset_env[f0:f1]))
        snapped.append(float(librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)))
    beat_times = np.unique(np.round(snapped, 4))

    # 4/4 正拍相位：按 onset 能量选 meter phase（StackOverflow / DJ grid 常用）
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    strengths = onset_env[frames]
    meter = 4
    n = (len(strengths) // meter) * meter
    if n >= meter:
        grid = strengths[:n].reshape(-1, meter)
        phase = int(np.argmax(grid.sum(axis=0)))
        downbeats = beat_times[phase::meter]
    else:
        downbeats = beat_times[::meter] if len(beat_times) else beat_times

    if len(beat_times) >= 2:
        tempo = float(60.0 / max(float(np.median(np.diff(beat_times))), 1e-6))
    return tempo, beat_times, np.asarray(downbeats, dtype=float)


def _madmom_beats(audio_path: str | Path) -> tuple[float, np.ndarray, np.ndarray]:
    from madmom_infer.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    act = RNNDownBeatProcessor()(str(audio_path))
    # 默认参数对齐 madmom / MIREX 常用设置（Böck ISMIR）
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=55,
        max_bpm=215,
        fps=100,
        correct=True,
    )
    out = np.asarray(dbn(act), dtype=float)
    if out.size == 0:
        raise RuntimeError("madmom_infer returned no beats")
    if out.ndim == 1:
        beats = out
        downs = out[:1]
    else:
        beats = out[:, 0]
        # 列1 为小节内拍号，1=正拍（downbeat）
        downs = out[out[:, 1] == 1, 0] if out.shape[1] > 1 else beats[::4]
    beats = np.asarray(beats, dtype=float)
    downs = np.asarray(downs, dtype=float)
    if len(downs) == 0:
        downs = beats[::4]
    tempo = float(60.0 / max(float(np.median(np.diff(beats))), 1e-6)) if len(beats) >= 2 else 120.0
    return tempo, beats, downs


def _onset_grid_score(
    onset_env: np.ndarray, times: np.ndarray, sr: int, hop_length: int, radius: int = 2
) -> float:
    if len(times) == 0:
        return 0.0
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    n = len(onset_env)
    vals = []
    for f in frames:
        lo = max(0, int(f) - radius)
        hi = min(n, int(f) + radius + 1)
        vals.append(float(np.max(onset_env[lo:hi])))
    return float(np.mean(vals))


def _double_beat_grid(beats: np.ndarray) -> np.ndarray:
    """在相邻拍中点插入一拍 → 节奏翻倍（octave up）。"""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 2:
        return beats
    mids = 0.5 * (beats[:-1] + beats[1:])
    return np.unique(np.round(np.concatenate([beats, mids]), 4))


def resolve_octave(
    y: np.ndarray,
    sr: int,
    beats: np.ndarray,
    downs: np.ndarray,
    tempo: float,
    hop_length: int = 512,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """
    动态 octave 选择（快歌半速锁 → 翻倍）。

    结合：
    - onset 网格得分（T vs 2T）
    - librosa 全局 tempo 提示（常更接近听感拍速）
    - tempogram 在 2T 处的相对能量
    """
    beats = np.asarray(beats, dtype=float)
    downs = np.asarray(downs, dtype=float)
    if len(beats) < 4:
        return tempo, beats, downs, "1x"

    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    tempo_hint = float(
        np.atleast_1d(
            librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop_length)
        )[0]
    )

    s1 = _onset_grid_score(onset, beats, sr, hop_length)
    beats2 = _double_beat_grid(beats)
    s2 = _onset_grid_score(onset, beats2, sr, hop_length)

    # tempogram：比较周期 τ 与 τ/2
    tg = np.abs(librosa.feature.tempogram(onset_envelope=onset, sr=sr, hop_length=hop_length))
    freqs = librosa.tempo_frequencies(tg.shape[0], hop_length=hop_length, sr=sr)
    # 取与 tempo / 2*tempo 最近的 bin 能量
    def _tg_at(bpm: float) -> float:
        if len(freqs) == 0:
            return 0.0
        i = int(np.argmin(np.abs(freqs - bpm)))
        return float(np.mean(tg[max(0, i - 1) : i + 2]))

    e1 = _tg_at(tempo)
    e2 = _tg_at(tempo * 2.0)

    choose_2x = False
    err1 = abs(np.log2(max(tempo, 1.0) / max(tempo_hint, 1.0)))
    err2 = abs(np.log2(max(tempo * 2.0, 1.0) / max(tempo_hint, 1.0)))

    # 1) librosa 听感拍速明显接近 2T（快歌半速锁的主判据）
    if tempo < 110 and tempo_hint >= tempo * 1.45 and err2 + 0.08 < err1:
        choose_2x = True
    # 2) 2T 网格 onset 更贴
    elif s2 > s1 * 1.05:
        choose_2x = True
    # 3) onset 接近时，tempogram / hint 倾向 2T
    elif s2 >= s1 * 0.90:
        if err2 + 0.10 < err1:
            choose_2x = True
        elif e2 > e1 * 1.10 and tempo_hint >= tempo * 1.35:
            choose_2x = True

    # 已经很快就不再翻倍
    if tempo * 2.0 > 200:
        choose_2x = False

    if choose_2x:
        # 中点轻吸到 onset，保持相位准
        med_ibi = float(np.median(np.diff(beats2))) if len(beats2) > 1 else 0.4
        win = float(np.clip(med_ibi * 0.10, 0.03, 0.08))
        duration = float(librosa.get_duration(y=y, sr=sr))
        snapped = []
        for t in beats2:
            f0 = int(librosa.time_to_frames(max(0.0, t - win), sr=sr, hop_length=hop_length))
            f1 = int(librosa.time_to_frames(min(duration, t + win), sr=sr, hop_length=hop_length))
            f0 = max(0, f0)
            f1 = min(len(onset), max(f0 + 1, f1))
            peak = f0 + int(np.argmax(onset[f0:f1]))
            snapped.append(float(librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)))
        beats2 = np.unique(np.round(snapped, 4))
        # 原 downs 仍是小节头；新拍格下每 8 拍一小节（旧 4 拍×2）
        if len(downs) == 0:
            downs = beats2[::8]
        return tempo * 2.0, beats2, downs, "2x"

    return tempo, beats, downs, "1x"


def extract_beats(
    y: np.ndarray,
    sr: int,
    path: str | None = None,
    hop_length: int = 512,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """
    Returns
    -------
    tempo, beat_times, downbeat_times, backend
    """
    tmp: str | None = None
    backend = "librosa"
    try:
        # madmom_infer 需要 44.1kHz wav 路径（不自动重采样）
        if path and Path(path).suffix.lower() in {".wav"} and sr == 44100:
            audio_path = path
        else:
            y44, _ = librosa.load(path, sr=44100, mono=True) if path else (None, None)
            if y44 is None:
                y44 = librosa.resample(np.asarray(y, dtype=float), orig_sr=sr, target_sr=44100)
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
            sf.write(tmp, y44, 44100)
            audio_path = tmp
        tempo, beats, downs = _madmom_beats(audio_path)
        backend = "madmom_infer"
    except Exception:
        tempo, beats, downs = _librosa_beats(y, sr, hop_length=hop_length)
        backend = "librosa"
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass

    tempo, beats, downs, octv = resolve_octave(
        y, sr, beats, downs, tempo, hop_length=hop_length
    )
    return tempo, beats, downs, f"{backend}:{octv}"