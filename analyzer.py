"""音乐结构 / 情绪 / 风格 — 分阶段分析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import librosa
import numpy as np

from emotion import (
    apply_section_intensity_contrast,
    contrast_stretch,
    intensity_raw,
    refine_result_emotions,
    tag_from_va,
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
    emotion_tag: str
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


def _emotion_curves(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, ...]:
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).mean(axis=0)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_major = chroma[[0, 2, 4, 5, 7, 9, 11], :].mean(axis=0)
    chroma_minor = chroma[[0, 2, 3, 5, 7, 8, 10], :].mean(axis=0)
    mode_bias = chroma_major - chroma_minor
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    energy = _smooth(_normalize(rms), 15)
    brightness = _smooth(_normalize(cent), 15)
    contrast_n = _normalize(contrast)
    onset_n = _normalize(_smooth(onset, 11))
    arousal = _smooth(0.40 * energy + 0.25 * contrast_n + 0.15 * _normalize(zcr) + 0.20 * onset_n, 15)
    valence = _smooth(0.40 * brightness + 0.45 * _normalize(mode_bias) + 0.15 * (1 - contrast_n), 15)
    raw_i = intensity_raw(energy, arousal, valence, onset_n)
    intensity = contrast_stretch(_smooth(np.asarray(raw_i), 11), lo_pct=5, hi_pct=95, gamma=0.68)
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

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo)[0])

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
                emotion_tag=tag_from_va(ar, va),
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
