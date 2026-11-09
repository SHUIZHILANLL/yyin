"""
音乐风格 / 流派估计

库与模型调研（Windows 兼容性）：
- Essentia Discogs-EffNet / genre_discogs400：Linux/macOS，无 Windows wheel
- madmom：依赖 Cython 编译，Windows 易失败
- musicnn / transformers Hubert-GTZAN：需 PyTorch，体量大
- 本模块：GTZAN 十类声学特征打分（MFCC / 频谱 / 节奏 / 和声复杂度）
  特征族对齐 GTZAN 常见管线（librosa + 手工特征），并输出次级风格标签

GTZAN 标签：blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np


GTZAN_LABELS_ZH = {
    "blues": "布鲁斯",
    "classical": "古典",
    "country": "乡村",
    "disco": "迪斯科",
    "hiphop": "嘻哈",
    "jazz": "爵士",
    "metal": "金属",
    "pop": "流行",
    "reggae": "雷鬼",
    "rock": "摇滚",
}


@dataclass
class StyleScore:
    label: str
    label_zh: str
    score: float


@dataclass
class StyleResult:
    top_style: str
    top_style_zh: str
    confidence: float
    scores: list[StyleScore]
    tags: list[str]
    descriptors: dict[str, float]
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(v: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(v)


def _extract_descriptors(y: np.ndarray, sr: int) -> dict[str, float]:
    hop = 512
    # 时长截断加速：最多用中间 90s
    max_len = int(90 * sr)
    if len(y) > max_len:
        mid = len(y) // 2
        y = y[mid - max_len // 2 : mid + max_len // 2]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # 节奏规则度：onset 自相关峰
    onset_n = onset / (np.max(onset) + 1e-8)
    ac = librosa.autocorrelate(onset_n, max_size=min(len(onset_n), sr // hop * 4))
    ac = ac[len(ac) // 20 :]  # 去掉 0 lag 附近
    rhythm_reg = float(np.max(ac) / (np.mean(ac) + 1e-8)) if len(ac) else 0.0

    chroma_entropy = float(-np.sum(np.mean(chroma, axis=1) * np.log(np.mean(chroma, axis=1) + 1e-8)))
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    h_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-8))
    p_ratio = float(np.mean(np.abs(percussive)) / (np.mean(np.abs(y)) + 1e-8))

    desc = {
        "tempo": _safe(tempo),
        "energy_mean": _safe(float(np.mean(rms))),
        "energy_std": _safe(float(np.std(rms))),
        "zcr_mean": _safe(float(np.mean(zcr))),
        "centroid_mean": _safe(float(np.mean(cent) / (sr / 2))),
        "rolloff_mean": _safe(float(np.mean(rolloff) / (sr / 2))),
        "bandwidth_mean": _safe(float(np.mean(bandwidth) / (sr / 2))),
        "contrast_mean": _safe(float(np.mean(contrast))),
        "flatness_mean": _safe(float(np.mean(flatness))),
        "mfcc1_mean": _safe(float(np.mean(mfcc[1]))),
        "mfcc2_mean": _safe(float(np.mean(mfcc[2]))),
        "chroma_entropy": _safe(chroma_entropy),
        "rhythm_regularity": _safe(min(rhythm_reg / 20.0, 1.5)),
        "harmonic_ratio": _safe(h_ratio),
        "percussive_ratio": _safe(p_ratio),
        "dynamic_range": _safe(float(np.percentile(rms, 90) - np.percentile(rms, 10))),
    }
    return desc


def _softmax(xs: dict[str, float], temp: float = 0.35) -> dict[str, float]:
    keys = list(xs.keys())
    arr = np.array([xs[k] for k in keys], dtype=float)
    arr = arr - np.max(arr)
    e = np.exp(arr / max(temp, 1e-3))
    e = e / (e.sum() + 1e-8)
    return {k: float(v) for k, v in zip(keys, e)}


def _score_genres(d: dict[str, float]) -> dict[str, float]:
    """
    基于 GTZAN 文献中常见可分特征的软打分（启发式原型匹配）。
    分数越高越像该类。
    """
    t = d["tempo"]
    e = d["energy_mean"]
    z = d["zcr_mean"]
    c = d["centroid_mean"]
    f = d["flatness_mean"]
    r = d["rhythm_regularity"]
    h = d["harmonic_ratio"]
    p = d["percussive_ratio"]
    ce = d["chroma_entropy"]
    dr = d["dynamic_range"]
    bw = d["bandwidth_mean"]
    contrast = d["contrast_mean"]

    scores = {
        # 古典：低打击、高谐波、低 ZCR、频谱较暗偏稳
        "classical": 1.4 * h + 0.6 * (1 - p) + 0.5 * (1 - z * 8) + 0.4 * (1 - f) + 0.3 * (1 - e * 3) - 0.2 * (abs(t - 90) / 90),
        # 爵士：和声复杂、中等能量、摇摆节奏不极规则
        "jazz": 0.9 * ce / 3 + 0.7 * h + 0.4 * (1 - abs(r - 0.6)) + 0.3 * dr * 5 + 0.2 * (1 - abs(t - 120) / 120),
        # 布鲁斯：中慢速、谐波、中能量
        "blues": 0.8 * h + 0.5 * (1 - abs(t - 90) / 90) + 0.4 * e * 2 + 0.3 * (1 - f) + 0.2 * ce / 3,
        # 乡村：中亮、中速、谐波为主
        "country": 0.7 * h + 0.5 * (1 - abs(t - 110) / 110) + 0.4 * c + 0.3 * (1 - p) + 0.2 * (1 - f),
        # 流行：中高能量、规则节奏、偏亮
        "pop": 0.8 * r + 0.7 * e * 3 + 0.5 * c + 0.4 * (1 - abs(t - 118) / 118) + 0.3 * p,
        # 摇滚：能量高、失真感(flatness/带宽)、中快
        "rock": 0.9 * e * 3 + 0.7 * p + 0.5 * bw + 0.4 * (1 - abs(t - 130) / 130) + 0.3 * f * 2,
        # 金属：很高能量/ZCR/亮度、快
        "metal": 1.1 * e * 3 + 0.9 * z * 10 + 0.8 * c + 0.6 * (t / 160) + 0.5 * f * 2 + 0.3 * contrast / 30,
        # 嘻哈：强打击、规则节奏、语速感(zcr)、常 85-100
        "hiphop": 1.0 * p + 0.9 * r + 0.6 * (1 - abs(t - 92) / 92) + 0.5 * z * 8 + 0.3 * e * 2,
        # 雷鬼：偏慢、重拍、节奏规则、中暗
        "reggae": 0.9 * r + 0.8 * (1 - abs(t - 75) / 75) + 0.5 * p + 0.4 * (1 - c) + 0.3 * e * 2,
        # 迪斯科：快、非常规则、亮、舞曲能量
        "disco": 1.0 * r + 0.8 * (1 - abs(t - 120) / 120) + 0.7 * e * 3 + 0.5 * c + 0.4 * p,
    }
    return scores


def _tags_from_desc(d: dict[str, float], top: str) -> list[str]:
    tags = [GTZAN_LABELS_ZH.get(top, top)]
    if d["tempo"] >= 130:
        tags.append("快节奏")
    elif d["tempo"] <= 85:
        tags.append("慢节奏")
    else:
        tags.append("中速")
    if d["energy_mean"] > 0.12:
        tags.append("高能量")
    elif d["energy_mean"] < 0.05:
        tags.append("低能量")
    if d["percussive_ratio"] > 0.55:
        tags.append("强节奏型")
    if d["harmonic_ratio"] > 0.55:
        tags.append("和声主导")
    if d["centroid_mean"] > 0.35:
        tags.append("音色明亮")
    elif d["centroid_mean"] < 0.18:
        tags.append("音色偏暗")
    if d["flatness_mean"] > 0.15:
        tags.append("噪声/失真感")
    if d["rhythm_regularity"] > 0.8:
        tags.append("节奏规整")
    if d["chroma_entropy"] > 2.2:
        tags.append("和声丰富")
    # 去重保序
    out = []
    for t in tags:
        if t not in out:
            out.append(t)
    return out[:8]


def _try_transformers_gtzan(path: str) -> StyleResult | None:
    """可选：HuggingFace distilhubert GTZAN 模型。"""
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except Exception:
        return None
    try:
        model_id = "SeyedAli/Musical-genres-Classification-Hubert-V1"
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        y, sr = librosa.load(path, sr=extractor.sampling_rate, mono=True, duration=30)
        inputs = extractor(y, sampling_rate=extractor.sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
        id2label = model.config.id2label
        pairs = sorted([(id2label[i].lower(), float(prob[i])) for i in range(len(prob))], key=lambda x: -x[1])
        scores = [
            StyleScore(label=lab, label_zh=GTZAN_LABELS_ZH.get(lab, lab), score=round(sc, 4))
            for lab, sc in pairs
        ]
        top, conf = pairs[0]
        y2, sr2 = librosa.load(path, sr=22050, mono=True, duration=60)
        desc = _extract_descriptors(y2, sr2)
        return StyleResult(
            top_style=top,
            top_style_zh=GTZAN_LABELS_ZH.get(top, top),
            confidence=round(conf, 4),
            scores=scores,
            tags=_tags_from_desc(desc, top),
            descriptors={k: round(v, 4) for k, v in desc.items()},
            backend=f"transformers:{model_id}",
        )
    except Exception:
        return None


def analyze_style(path: str, y: np.ndarray | None = None, sr: int = 22050) -> StyleResult:
    deep = _try_transformers_gtzan(path)
    if deep is not None:
        return deep

    if y is None:
        y, sr = librosa.load(path, sr=sr, mono=True)
    desc = _extract_descriptors(y, sr)
    raw = _score_genres(desc)
    prob = _softmax(raw, temp=0.45)
    ordered = sorted(prob.items(), key=lambda x: -x[1])
    scores = [
        StyleScore(label=k, label_zh=GTZAN_LABELS_ZH.get(k, k), score=round(v, 4))
        for k, v in ordered
    ]
    top, conf = ordered[0]
    return StyleResult(
        top_style=top,
        top_style_zh=GTZAN_LABELS_ZH.get(top, top),
        confidence=round(conf, 4),
        scores=scores,
        tags=_tags_from_desc(desc, top),
        descriptors={k: round(v, 4) for k, v in desc.items()},
        backend="librosa-gtzan-features",
    )
