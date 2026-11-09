"""
情绪指数（效价 / 唤醒 / 强度）稳定化

参考：
- MediaEval Emotion in Music / DEAM：连续 VA 预测后做 moving-average 平滑
  （Pellegrini 2015：约 13 点滑窗；SAIL 2014：arousal≈6.5s、valence 更长）
- TUM MediaEval 2013：特征/目标用标准化或压到有限区间，避免爆炸尺度
- 相对表征（Saari et al.）：相对全曲均值，但用 tanh 软压缩，不做强制排名拉满

原则：保留段间可辨差异，抑制「最低 0.08 / 最高 0.95」式剧烈跳变。
"""

from __future__ import annotations

import numpy as np

from style import StyleResult


STYLE_PRIOR = {
    "metal": (0.18, -0.06),
    "rock": (0.10, 0.02),
    "disco": (0.12, 0.08),
    "hiphop": (0.08, 0.00),
    "pop": (0.06, 0.06),
    "reggae": (0.02, 0.04),
    "country": (0.00, 0.05),
    "blues": (-0.02, -0.04),
    "jazz": (0.02, 0.03),
    "classical": (-0.08, 0.04),
}


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """两端保留，中间滑窗平均（MediaEval 常用后处理）。"""
    x = np.asarray(x, dtype=float)
    if win < 3 or len(x) < win:
        return x
    win = win | 1
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(xp, kernel, mode="valid")


def soft_normalize_curve(x: np.ndarray, gain: float = 0.65) -> np.ndarray:
    """
    曲内软标准化 → (0,1)：z-score + tanh，再映到 0.5 中心。
    gain≈0.55–0.70：|z|=2 大约落在 0.25–0.75，避免尖峰吃满量程。
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-6
    z = (x - mu) / sd
    return 0.5 + 0.5 * np.tanh(gain * z)


def shrink_to_mean(x: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """向曲/段均值收缩：strength=0 不变，1 全压到均值。"""
    x = np.asarray(x, dtype=float)
    strength = float(np.clip(strength, 0, 1))
    mu = float(np.mean(x))
    return mu + (1.0 - strength) * (x - mu)


def intensity_raw(
    energy: np.ndarray | float,
    arousal: np.ndarray | float,
    valence: np.ndarray | float,
    onset: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """强度原值：能量主导，onset/唤醒次之（Yang & Chen activation↔energy）。"""
    e = np.asarray(energy, dtype=float)
    a = np.asarray(arousal, dtype=float)
    v = np.asarray(valence, dtype=float)
    if onset is None:
        o = 0.5 * a
    else:
        o = np.clip(np.asarray(onset, dtype=float), 0, 1)
    charge = np.abs(v - 0.5) * 2.0
    return 0.50 * e + 0.28 * a + 0.15 * o + 0.07 * charge


def soft_section_scale(values: np.ndarray, gain: float = 0.45) -> np.ndarray:
    """
    段间软相对缩放：tanh(gain·z)。
    gain≈0.4–0.5 → 典型段差约 0.25–0.40，不再拉满 0.08–0.95。
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    if len(arr) == 1:
        return np.array([0.5])
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-6:
        return np.full_like(arr, 0.5, dtype=float)
    z = (arr - mu) / sd
    return 0.5 + 0.5 * np.tanh(gain * z)


def blend_absolute_relative(
    absolute: np.ndarray,
    relative: np.ndarray,
    abs_w: float = 0.70,
    shrink: float = 0.35,
) -> np.ndarray:
    """绝对水平（保留 Intro 偏静）与相对对比混合，再向心收缩。"""
    abs_w = float(np.clip(abs_w, 0, 1))
    abs_c = shrink_to_mean(np.clip(absolute, 0, 1), strength=shrink)
    out = abs_w * abs_c + (1.0 - abs_w) * relative
    return np.clip(out, 0.12, 0.88)


def build_stable_curves(
    energy: np.ndarray,
    arousal: np.ndarray,
    valence: np.ndarray,
    onset: np.ndarray,
    sr_frames_per_sec: float = 22050 / 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    帧级稳定曲线：先长窗平滑，再软归一。
    arousal ~6–7s，valence ~12–15s，intensity 介于两者（DEAM/SAIL 经验）。
    """
    # DEAM：arousal≈6.5s、valence≈19s；intensity 略短窗，保留一点起伏
    win_a = max(5, int(round(6.5 * sr_frames_per_sec)) | 1)
    win_v = max(5, int(round(19.0 * sr_frames_per_sec)) | 1)
    win_i = max(5, int(round(5.0 * sr_frames_per_sec)) | 1)
    win_e = max(5, int(round(6.0 * sr_frames_per_sec)) | 1)

    e_s = moving_average(energy, win_e)
    a_s = moving_average(arousal, win_a)
    v_s = moving_average(valence, win_v)
    o_s = moving_average(onset, win_a)

    e_n = soft_normalize_curve(e_s, gain=0.60)
    a_n = soft_normalize_curve(a_s, gain=0.55)
    v_n = soft_normalize_curve(v_s, gain=0.50)
    # onset 少平滑一点，给强度曲线更多瞬时起伏
    o_n = soft_normalize_curve(moving_average(onset, max(5, win_i)), gain=0.65)
    raw_i = intensity_raw(e_n, a_n, v_n, o_n)
    i_s = moving_average(np.asarray(raw_i, dtype=float), win_i)
    i_n = soft_normalize_curve(i_s, gain=0.72)
    return e_n, a_n, v_n, i_n


def differentiate_sections(result, style: StyleResult | None = None) -> None:
    """段级指数：曲线均值 + 温和相对缩放（不再强制排名拉满）。"""
    if not result.sections:
        return

    e0, a0, v0, i0 = [], [], [], []
    for sec in result.sections:
        i_a = int(np.searchsorted(result.times, sec.start, side="left"))
        i_b = int(np.searchsorted(result.times, sec.end, side="right"))
        i_b = max(i_a + 1, min(i_b, len(result.times)))
        sl = slice(i_a, i_b)
        e0.append(float(np.mean(result.energy_curve[sl])))
        a0.append(float(np.mean(result.arousal_curve[sl])))
        v0.append(float(np.mean(result.valence_curve[sl])))
        i0.append(float(np.mean(result.intensity_curve[sl])))

    e0 = np.asarray(e0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    i0 = np.asarray(i0, dtype=float)

    if style is not None:
        ab0, vb0 = STYLE_PRIOR.get(style.top_style, (0.0, 0.0))
        w = 0.15 * float(np.clip(style.confidence, 0.1, 0.5))
        a0 = np.clip(a0 + ab0 * w * 0.5, 0, 1)
        v0 = np.clip(v0 + vb0 * w * 0.5, 0, 1)
        i0 = np.clip(i0 + ab0 * w * 0.25, 0, 1)

    e1 = blend_absolute_relative(e0, soft_section_scale(e0, gain=0.45), abs_w=0.72, shrink=0.40)
    a1 = blend_absolute_relative(a0, soft_section_scale(a0, gain=0.40), abs_w=0.72, shrink=0.40)
    v1 = blend_absolute_relative(v0, soft_section_scale(v0, gain=0.35), abs_w=0.75, shrink=0.45)
    i1 = blend_absolute_relative(i0, soft_section_scale(i0, gain=0.48), abs_w=0.68, shrink=0.38)

    for k, sec in enumerate(result.sections):
        sec.energy = round(float(e1[k]), 3)
        sec.arousal = round(float(a1[k]), 3)
        sec.valence = round(float(v1[k]), 3)
        sec.intensity = round(float(i1[k]), 3)


def apply_section_intensity_contrast(result) -> None:
    differentiate_sections(result, style=getattr(result, "style", None))


def refine_result_emotions(result) -> None:
    """风格后轻微偏置 + 重算稳定强度曲线 + 段级软缩放。"""
    style = getattr(result, "style", None)
    hop = 512
    fps = float(result.sr) / hop

    onset_proxy = np.clip(
        0.55 * result.arousal_curve + 0.45 * moving_average(
            np.abs(np.gradient(result.energy_curve)) * 6.0, max(5, int(fps * 2) | 1)
        ),
        0,
        1,
    )
    if style is not None:
        ab0, vb0 = STYLE_PRIOR.get(style.top_style, (0.0, 0.0))
        w = float(np.clip(style.confidence, 0.1, 0.3))
        result.arousal_curve = np.clip(result.arousal_curve + ab0 * w * 0.25, 0, 1)
        result.valence_curve = np.clip(result.valence_curve + vb0 * w * 0.25, 0, 1)

    e_n, a_n, v_n, i_n = build_stable_curves(
        result.energy_curve,
        result.arousal_curve,
        result.valence_curve,
        onset_proxy,
        sr_frames_per_sec=fps,
    )
    # 与已有曲线轻度混合，避免整页视觉突变
    result.energy_curve = 0.35 * result.energy_curve + 0.65 * e_n
    result.arousal_curve = 0.35 * result.arousal_curve + 0.65 * a_n
    result.valence_curve = 0.40 * result.valence_curve + 0.60 * v_n
    result.intensity_curve = i_n

    differentiate_sections(result, style=style)


# 兼容旧导入名
def contrast_stretch(x: np.ndarray, lo_pct: float = 8.0, hi_pct: float = 92.0, gamma: float = 0.72) -> np.ndarray:
    """兼容旧名：改为软归一（忽略 percentile/gamma 强拉伸参数）。"""
    return soft_normalize_curve(np.asarray(x, dtype=float), gain=0.55)
