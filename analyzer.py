"""音乐结构 / 情绪 / 风格 — 分阶段分析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import librosa
import numpy as np

from emotion import (
    apply_section_intensity_contrast,
    build_stable_curves,
    intensity_raw,
    moving_average,
    refine_result_emotions,
    soft_normalize_curve,
)
from structure import analyze_structure
from style import StyleResult, analyze_style

Stage = Literal["core", "style", "all"]


@dataclass
class Section:
    index: int
    start: float
    end: float
    label: str
    duration: float
    energy: float
    arousal: float
    valence: float
    intensity: float
    tempo_local: float
    brightness: float
    spectral_contrast: float
    cluster_id: int = 0


@dataclass
class AnalysisResult:
    path: str
    duration: float
    sr: int
    tempo: float
    sections: list[Section]
    times: np.ndarray
    energy_curve: np.ndarray
    arousal_curve: np.ndarray
    valence_curve: np.ndarray
    intensity_curve: np.ndarray
    waveform: np.ndarray
    waveform_sr: int
    novelty: np.ndarray
    novelty_times: np.ndarray
    beat_times: np.ndarray | None = None
    method: str = "scluster"
    style: StyleResult | None = None
    _y: np.ndarray | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": round(float(self.duration), 2),
            "tempo": round(float(self.tempo), 1),
            "method": self.method,
            "section_count": len(self.sections),
            "sections": [asdict(s) for s in self.sections],
            "style": self.style.to_dict() if self.style else None,
        }


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _smooth(x: np.ndarray, win: int = 9) -> np.ndarray:
    if len(x) < 3:
        return x
    win = max(3, win | 1)
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def _beat_accent_score(
    onset_env: np.ndarray, times: np.ndarray, sr: int, hop_length: int, radius: int = 2
) -> float:
    """拍点邻域 onset 峰值均值（SO/Ellis：主力度常在检测点附近而非精确落点）。"""
    if len(times) == 0:
        return 0.0
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    scores = []
    n = len(onset_env)
    for f in frames:
        lo = max(0, int(f) - radius)
        hi = min(n, int(f) + radius + 1)
        scores.append(float(np.max(onset_env[lo:hi])))
    return float(np.mean(scores))


def _extract_stable_beats(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> tuple[float, np.ndarray]:
    """
    拍点网格（参考 librosa/Ellis DP + StackOverflow 正拍相位做法）：
    1. onset_strength(aggregate=median) → 同一包络送入 beat_track
    2. 时变 tempo（自由拍/局部变速）作为 bpm 先验
    3. 倍速 octave error：按 accent 选奇/偶子集
    4. 全局半拍相位二选一（正拍 vs 反拍），只校正一次，禁止分段乱移格
    """
    duration = float(librosa.get_duration(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    # 局部拍速先验：比全局常数 BPM 更跟自由拍（librosa 支持 bpm 与 onset 同长）
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
    beat_times = beat_times[(beat_times >= 0.0) & (beat_times <= duration + 1e-6)]
    tempo = float(np.median(np.atleast_1d(tempo_arr).astype(float)))
    if len(beat_times) < 2:
        return tempo, beat_times

    med_ibi = float(np.median(np.diff(beat_times)))
    local_bpm = 60.0 / max(med_ibi, 1e-6)

    # Octave / double-tempo：隔拍取能量更高的一套（正拍集合）
    if local_bpm > tempo * 1.20 and len(beat_times) >= 4:
        even, odd = beat_times[0::2], beat_times[1::2]
        s_even = _beat_accent_score(onset_env, even, sr, hop_length)
        s_odd = _beat_accent_score(onset_env, odd, sr, hop_length)
        beat_times = even if s_even >= s_odd else odd
        if len(beat_times) >= 2:
            med_ibi = float(np.median(np.diff(beat_times)))
            tempo = 60.0 / max(med_ibi, 1e-6)

    # 正拍/反拍：整格平移半拍，只做一次全局选择（避免分段校正把格子拧乱）
    score0 = _beat_accent_score(onset_env, beat_times, sr, hop_length)
    shifted = beat_times + med_ibi * 0.5
    shifted = shifted[(shifted >= 0.0) & (shifted <= duration)]
    score1 = _beat_accent_score(onset_env, shifted, sr, hop_length)
    if score1 > score0 * 1.05:
        beat_times = shifted
        score0 = score1

    # 小范围全局微移（±15% 拍距），锁到 onset 峰心；不做逐拍乱吸附
    best_shift, best_score = 0.0, score0
    for k in range(13):
        shift = ((k / 12.0) - 0.5) * med_ibi * 0.30
        cand = beat_times + shift
        cand = cand[(cand >= 0.0) & (cand <= duration)]
        sc = _beat_accent_score(onset_env, cand, sr, hop_length)
        if sc > best_score:
            best_score, best_shift = sc, shift
    beat_times = beat_times + best_shift
    beat_times = beat_times[(beat_times >= 0.0) & (beat_times <= duration)]
    beat_times = np.unique(np.round(beat_times, 4))

    if len(beat_times) >= 2:
        tempo = float(60.0 / max(float(np.median(np.diff(beat_times))), 1e-6))
    return tempo, beat_times.astype(float)


def _emotion_curves(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, ...]:
    """
    连续情绪曲线：MediaEval/DEAM 风格后处理
    - arousal 滑窗 ≈6.5s，valence ≈14–19s（SAIL 2015: MA 13 / 38 @2Hz）
    - 软归一（z→tanh）代替 percentile gamma 强拉伸
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).mean(axis=0)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_major = chroma[[0, 2, 4, 5, 7, 9, 11], :].mean(axis=0)
    chroma_minor = chroma[[0, 2, 3, 5, 7, 8, 10], :].mean(axis=0)
    mode_bias = chroma_major - chroma_minor
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    fps = float(sr) / hop_length
    win_e = max(5, int(round(6.0 * fps)) | 1)
    win_a = max(5, int(round(6.5 * fps)) | 1)
    win_v = max(5, int(round(19.0 * fps)) | 1)

    energy_raw = moving_average(_normalize(rms), win_e)
    brightness = moving_average(_normalize(cent), win_e)
    contrast_n = soft_normalize_curve(moving_average(contrast, win_e), gain=0.55)
    onset_n = soft_normalize_curve(moving_average(onset, win_a), gain=0.55)
    zcr_n = soft_normalize_curve(moving_average(zcr, win_e), gain=0.50)
    mode_n = soft_normalize_curve(moving_average(mode_bias, win_v), gain=0.45)

    energy = soft_normalize_curve(energy_raw, gain=0.60)
    arousal = soft_normalize_curve(
        moving_average(0.40 * energy + 0.25 * contrast_n + 0.15 * zcr_n + 0.20 * onset_n, win_a),
        gain=0.55,
    )
    valence = soft_normalize_curve(
        moving_average(0.40 * brightness + 0.45 * mode_n + 0.15 * (1.0 - contrast_n), win_v),
        gain=0.50,
    )
    _, _, _, intensity = build_stable_curves(
        energy, arousal, valence, onset_n, sr_frames_per_sec=fps
    )
    # 掺一点短窗 raw，曲线多一点起伏，但仍压尖峰
    win_i_short = max(5, int(round(3.0 * fps)) | 1)
    raw_i = np.asarray(intensity_raw(energy, arousal, valence, onset_n), dtype=float)
    raw_i = soft_normalize_curve(moving_average(raw_i, win_i_short), gain=0.70)
    intensity = 0.62 * intensity + 0.38 * raw_i
    intensity = moving_average(intensity, max(5, win_i_short))

    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    return times, energy, brightness, contrast_n, arousal, valence, intensity, onset_n


def analyze_core(
    path: str,
    sr: int = 22050,
    n_sections: int | None = None,
    hop_length: int = 512,
    method: str = "scluster",
) -> AnalysisResult:
    """阶段1：结构分段 + 声学情绪（先出结果）。"""
    y, _ = librosa.load(path, sr=sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    tempo, beat_times = _extract_stable_beats(y, sr=sr, hop_length=hop_length)

    struct = analyze_structure(
        y, sr, method=method, n_sections=n_sections, hop_length=hop_length
    )
    times, energy, brightness, contrast, arousal, valence, intensity, onset_env = _emotion_curves(
        y, sr, hop_length
    )

    sections: list[Section] = []
    for i, name in enumerate(struct.section_names):
        start_t = float(struct.bound_times[i])
        end_t = float(struct.bound_times[i + 1])
        a = int(np.searchsorted(times, start_t, side="left"))
        b = int(np.searchsorted(times, end_t, side="right"))
        b = max(a + 1, min(b, len(times)))
        sl = slice(a, b)
        e = float(np.mean(energy[sl]))
        ar = float(np.mean(arousal[sl]))
        va = float(np.mean(valence[sl]))
        it = float(np.mean(intensity[sl]))
        br = float(np.mean(brightness[sl]))
        sc = float(np.mean(contrast[sl]))
        local_boost = float(np.mean(onset_env[a:b])) if b <= len(onset_env) else 0.5
        tempo_local = tempo * (0.85 + 0.30 * local_boost)
        cid = int(struct.segment_labels[i]) if i < len(struct.segment_labels) else 0
        sections.append(
            Section(
                index=i + 1,
                start=round(start_t, 2),
                end=round(end_t, 2),
                label=name,
                duration=round(end_t - start_t, 2),
                energy=round(e, 3),
                arousal=round(ar, 3),
                valence=round(va, 3),
                intensity=round(it, 3),
                tempo_local=round(tempo_local, 1),
                brightness=round(br, 3),
                spectral_contrast=round(sc, 3),
                cluster_id=cid,
            )
        )

    target_wave = min(len(y), 20000)
    wave_idx = np.linspace(0, len(y) - 1, target_wave).astype(int)

    result = AnalysisResult(
        path=path,
        duration=duration,
        sr=sr,
        tempo=tempo,
        sections=sections,
        times=times,
        energy_curve=energy,
        arousal_curve=arousal,
        valence_curve=valence,
        intensity_curve=intensity,
        waveform=y[wave_idx],
        waveform_sr=sr,
        novelty=struct.novelty,
        novelty_times=struct.novelty_times,
        beat_times=np.asarray(beat_times, dtype=float),
        method=struct.method,
        style=None,
        _y=y,
    )
    apply_section_intensity_contrast(result)
    return result


def analyze_style_stage(result: AnalysisResult) -> AnalysisResult:
    """阶段2：风格，并校正情绪。"""
    y = result._y
    if y is None:
        y, _ = librosa.load(result.path, sr=result.sr, mono=True)
        result._y = y
    result.style = analyze_style(result.path, y=y, sr=result.sr)
    refine_result_emotions(result)
    return result


def analyze_music(
    path: str,
    sr: int = 22050,
    n_sections: int | None = None,
    hop_length: int = 512,
    method: str = "scluster",
    stages: Stage | list[Stage] = "all",
) -> AnalysisResult:
    if stages == "all":
        want = ["core", "style"]
    elif isinstance(stages, str):
        want = [stages]
    else:
        want = list(stages)

    result = analyze_core(path, sr=sr, n_sections=n_sections, hop_length=hop_length, method=method)
    if "style" in want or stages == "all":
        analyze_style_stage(result)
    return result


def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"
