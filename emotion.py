"""
情绪标签（Russell 效价-唤醒）+ 段间相对评分

关键：同一首歌各段若都用绝对阈值，标签/强度会挤在一起。
做法：先算段内原始均值，再在「段与段」之间做相对拉开（排名 + 强制展宽），
情绪标签也按相对 VA 落象限，保证 Intro / Verse / Chorus 彼此可辨。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from style import StyleResult


EMOTION_QUADRANTS = [
    (0.66, 1.01, 0.58, 1.01, "兴奋欢快"),
    (0.66, 1.01, 0.42, 0.58, "紧张激动"),
    (0.66, 1.01, 0.00, 0.42, "愤怒焦虑"),
    (0.40, 0.66, 0.58, 1.01, "明朗轻快"),
    (0.40, 0.66, 0.42, 0.58, "平稳中性"),
    (0.40, 0.66, 0.00, 0.42, "忧伤沉思"),
    (0.00, 0.40, 0.58, 1.01, "宁静满足"),
    (0.00, 0.40, 0.42, 0.58, "安静过渡"),
    (0.00, 0.40, 0.00, 0.42, "低沉忧郁"),
]

STYLE_PRIOR = {
    "metal": (0.22, -0.08),
    "rock": (0.14, 0.02),
    "disco": (0.16, 0.12),
    "hiphop": (0.12, 0.00),
    "pop": (0.08, 0.10),
    "reggae": (0.02, 0.06),
    "country": (0.00, 0.08),
    "blues": (-0.02, -0.06),
    "jazz": (0.02, 0.04),
    "classical": (-0.10, 0.06),
}


@dataclass
class EmotionEstimate:
    arousal: float
    valence: float
    intensity: float
    tag: str


def tag_from_va(arousal: float, valence: float) -> str:
    a = float(np.clip(arousal, 0, 0.999))
    v = float(np.clip(valence, 0, 0.999))
    for a0, a1, v0, v1, tag in EMOTION_QUADRANTS:
        if a0 <= a < a1 and v0 <= v < v1:
            return tag
    return "平稳中性"


def contrast_stretch(
    x: np.ndarray,
    lo_pct: float = 8.0,
    hi_pct: float = 92.0,
    gamma: float = 0.72,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))
    if hi - lo < 1e-8:
        return np.clip(x, 0, 1)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return np.power(y, gamma)


def intensity_raw(
    energy: np.ndarray | float,
    arousal: np.ndarray | float,
    valence: np.ndarray | float,
    onset: np.ndarray | float | None = None,
) -> np.ndarray | float:
    e = np.asarray(energy, dtype=float)
    a = np.asarray(arousal, dtype=float)
    v = np.asarray(valence, dtype=float)
    e_p = np.power(np.clip(e, 0, 1), 0.85)
    a_p = np.power(np.clip(a, 0, 1), 0.90)
    charge = np.abs(v - 0.5) * 2.0
    if onset is None:
        o = 0.35 * a_p
    else:
        o = np.power(np.clip(np.asarray(onset, dtype=float), 0, 1), 0.90)
    return 0.48 * e_p + 0.27 * a_p + 0.18 * o + 0.07 * charge


def _spread_among_sections(values: list[float] | np.ndarray, power: float = 1.25) -> np.ndarray:
    """
    段间强制展宽：保留排序，把最低段→~0.08、最高段→~0.95，
    中间按排名插值并加一点原值形状，避免各段分数挤在一起。
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    if n == 1:
        return np.array([0.5])

    # 排名 0..n-1（并列取平均）
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # 处理并列
    for v in np.unique(arr):
        idx = np.where(arr == v)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))

    rank01 = ranks / (n - 1)
    # 原值 min-max
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        shape = np.full(n, 0.5)
    else:
        shape = (arr - lo) / (hi - lo)

    # 排名主导（拉开段差）+ 少量形状
    mixed = 0.75 * rank01 + 0.25 * shape
    mixed = np.power(np.clip(mixed, 0, 1), 1.0 / max(power, 1e-3))
    return 0.08 + 0.87 * mixed


def _ensure_tag_diversity(arousals: np.ndarray, valences: np.ndarray, tags: list[str]) -> list[str]:
    """
    若多段标签完全相同，按相对唤醒把最高/最低段改成更极端象限标签。
    """
    if len(tags) < 2:
        return tags
    if len(set(tags)) >= min(3, len(tags)):
        return tags
    out = list(tags)
    i_hi = int(np.argmax(arousals))
    i_lo = int(np.argmin(arousals))
    if arousals[i_hi] - arousals[i_lo] < 0.05:
        return out
    # 高唤醒段
    if valences[i_hi] >= 0.5:
        out[i_hi] = "兴奋欢快"
    else:
        out[i_hi] = "紧张激动"
    # 低唤醒段
    if valences[i_lo] >= 0.5:
        out[i_lo] = "宁静满足"
    else:
        out[i_lo] = "低沉忧郁"
    # 中间偏高/偏低再分一档
    if len(out) >= 3:
        mid_idx = [i for i in range(len(out)) if i not in (i_hi, i_lo)]
        for i in mid_idx:
            if arousals[i] >= 0.55 and out[i] in ("平稳中性", "安静过渡"):
                out[i] = "明朗轻快" if valences[i] >= 0.5 else "忧伤沉思"
    return out


def differentiate_sections(result, style: StyleResult | None = None) -> None:
    """
    核心：按「段与段」相对关系重写 energy/arousal/valence/intensity 与情绪标签。
    """
    if not result.sections:
        return

    n = len(result.sections)
    e0 = np.array([s.energy for s in result.sections], dtype=float)
    a0 = np.array([s.arousal for s in result.sections], dtype=float)
    v0 = np.array([s.valence for s in result.sections], dtype=float)
    # 原始强度：偏能量，段间才有差
    onset_means = []
    for sec in result.sections:
        i0 = int(np.searchsorted(result.times, sec.start, side="left"))
        i1 = int(np.searchsorted(result.times, sec.end, side="right"))
        i1 = max(i0 + 1, min(i1, len(result.energy_curve)))
        # 用能量梯度近似驱动
        g = np.abs(np.gradient(result.energy_curve[i0:i1])).mean() if i1 > i0 else 0.0
        onset_means.append(float(np.clip(g * 10, 0, 1)))
    o0 = np.asarray(onset_means, dtype=float)
    i_raw = np.asarray(intensity_raw(e0, a0, v0, o0), dtype=float)

    # 风格只微调相对效价/唤醒的「方向」，不做整曲平移（平移会抹平段差）
    if style is not None:
        ab0, vb0 = STYLE_PRIOR.get(style.top_style, (0.0, 0.0))
        w = 0.25 * float(np.clip(style.confidence, 0.1, 0.6))
        a0 = a0 + ab0 * w * (a0 - a0.mean() + 0.15)  # 高唤醒段多抬一点
        v0 = v0 + vb0 * w

    e1 = _spread_among_sections(e0, power=1.35)
    a1 = _spread_among_sections(a0, power=1.30)
    v1 = _spread_among_sections(v0, power=1.15)
    # 强度：能量主导 + 唤醒，再段间排名展宽
    i_seed = 0.62 * e0 + 0.28 * a0 + 0.10 * i_raw
    i1 = _spread_among_sections(i_seed, power=1.45)

    tags = [tag_from_va(float(a1[k]), float(v1[k])) for k in range(n)]
    tags = _ensure_tag_diversity(a1, v1, tags)

    for k, sec in enumerate(result.sections):
        sec.energy = round(float(e1[k]), 3)
        sec.arousal = round(float(a1[k]), 3)
        sec.valence = round(float(v1[k]), 3)
        sec.intensity = round(float(i1[k]), 3)
        sec.emotion_tag = tags[k]


def apply_section_intensity_contrast(result) -> None:
    """兼容旧名：完整段间分化。"""
    differentiate_sections(result, style=getattr(result, "style", None))


def refine_result_emotions(result) -> None:
    """风格分析后：轻量校正曲线，并再次做段间分化。"""
    style = getattr(result, "style", None)

    # 曲线仍用帧级拉伸，便于可视化
    onset_proxy = np.clip(
        0.6 * result.arousal_curve + 0.4 * np.abs(np.gradient(result.energy_curve)) * 8.0,
        0,
        1,
    )
    if style is not None:
        ab0, vb0 = STYLE_PRIOR.get(style.top_style, (0.0, 0.0))
        w = float(np.clip(style.confidence, 0.1, 0.35))
        # 只做轻微全局偏置，避免毁掉曲线起伏
        result.arousal_curve = np.clip(result.arousal_curve + ab0 * w * 0.35, 0, 1)
        result.valence_curve = np.clip(result.valence_curve + vb0 * w * 0.35, 0, 1)

    raw_curve = intensity_raw(
        result.energy_curve, result.arousal_curve, result.valence_curve, onset_proxy
    )
    result.intensity_curve = contrast_stretch(np.asarray(raw_curve), lo_pct=5, hi_pct=95, gamma=0.68)

    # 段级：先恢复「曲线均值」作种子，再段间拉开
    for sec in result.sections:
        a = int(np.searchsorted(result.times, sec.start, side="left"))
        b = int(np.searchsorted(result.times, sec.end, side="right"))
        b = max(a + 1, min(b, len(result.times)))
        sl = slice(a, b)
        sec.energy = float(np.mean(result.energy_curve[sl]))
        sec.arousal = float(np.mean(result.arousal_curve[sl]))
        sec.valence = float(np.mean(result.valence_curve[sl]))
        sec.intensity = float(np.mean(result.intensity_curve[sl]))

    differentiate_sections(result, style=style)
