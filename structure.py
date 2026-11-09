"""
音乐结构分析（边界 + 段标签）

参考实现 / 论文：
1. Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio Novelty. ICME.
   — 自相似矩阵 + 高斯棋盘核 → 新奇度峰值 = 边界
   — 参考: urinieto/msaf  algorithms/foote
2. McFee, B. & Ellis, D. (2014). Analyzing Song Structure with Spectral Clustering. ISMIR.
   — beat 同步 chroma+MFCC，递推亲和 + 路径相似度 → 归一化 Laplacian 谱嵌入 → KMeans
   — 参考: urinieto/msaf  algorithms/scluster/main2.py (Brian McFee)
3. FMP / AudioLabs C4S4 Novelty Segmentation
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.sparse.csgraph
from scipy.spatial import distance
from sklearn.cluster import KMeans


@dataclass
class StructureResult:
    bound_frames: np.ndarray          # 帧索引（含 0 与末帧）
    bound_times: np.ndarray           # 秒
    segment_labels: np.ndarray        # 每段聚类标签 0..K-1
    section_names: list[str]          # Intro/Verse/Chorus/...
    novelty: np.ndarray               # 帧级新奇度（展示用）
    novelty_times: np.ndarray
    method: str
    n_clusters: int


# --------------- Foote novelty (MSAF) ---------------

def _median_filter_feats(X: np.ndarray, M: int = 8) -> np.ndarray:
    """X: (n_frames, n_feats)"""
    out = X.copy()
    for i in range(out.shape[1]):
        out[:, i] = scipy.ndimage.median_filter(out[:, i], size=M)
    return out


def _gaussian_checkerboard(M: int) -> np.ndarray:
    if M % 2 == 1:
        M += 1
    g = scipy.signal.windows.gaussian(M, M / 3.0, sym=True)
    G = np.outer(g, g)
    half = M // 2
    G[half:, :half] *= -1
    G[:half, half:] *= -1
    return G


def _ssm_cosine(X: np.ndarray) -> np.ndarray:
    """特征行向量 → 余弦自相似矩阵，值越大越相似。"""
    # 归一化后点积 = cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    S = Xn @ Xn.T
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return S


def _ssm_seuclidean(X: np.ndarray) -> np.ndarray:
    D = distance.pdist(X, metric="seuclidean")
    D = distance.squareform(D)
    D = D / (D.max() + 1e-8)
    return 1.0 - D


def foote_novelty(
    X: np.ndarray,
    M_gaussian: int = 64,
    m_median: int = 8,
) -> np.ndarray:
    """
    Foote 新奇度曲线。
    X: (n_frames, n_feats)  — 建议 beat 同步 chroma(+mfcc)
    """
    if X.shape[0] < 16:
        return np.zeros(X.shape[0])
    F = _median_filter_feats(X, M=m_median)
    # 控制 SSM 规模
    max_n = 800
    step = max(1, F.shape[0] // max_n)
    Fs = F[::step]
    S = _ssm_seuclidean(Fs)
    M = min(M_gaussian, max(16, (Fs.shape[0] // 6) * 2))
    if M % 2 == 1:
        M += 1
    G = _gaussian_checkerboard(M)
    half = M // 2
    nc = np.zeros(Fs.shape[0])
    for i in range(half, Fs.shape[0] - half):
        block = S[i - half : i + half, i - half : i + half]
        nc[i] = float(np.sum(block * G))
    nc = nc - nc.min()
    if nc.max() > 0:
        nc /= nc.max()
    # 插值回原长度
    src = np.linspace(0, 1, len(nc))
    dst = np.linspace(0, 1, X.shape[0])
    return np.interp(dst, src, nc)


def foote_peaks(nc: np.ndarray, L: int = 16, min_peak_dist: int = 8) -> np.ndarray:
    """MSAF 自适应阈值检峰。"""
    if len(nc) < 8:
        return np.array([], dtype=int)
    offset = float(nc.mean()) / 20.0
    smooth = scipy.ndimage.gaussian_filter1d(nc, sigma=4)
    th = scipy.ndimage.median_filter(smooth, size=max(3, L)) + offset
    peaks = []
    last = -10**9
    for i in range(1, len(smooth) - 1):
        if smooth[i - 1] < smooth[i] and smooth[i] > smooth[i + 1] and smooth[i] > th[i]:
            if i - last >= min_peak_dist:
                peaks.append(i)
                last = i
    return np.asarray(peaks, dtype=int)


# --------------- McFee Laplacian spectral clustering ---------------

def _embed_laplacian(
    A_rep: np.ndarray,
    A_loc: np.ndarray,
    rec_width: int = 9,
    rec_smooth: int = 9,
    evec_smooth: int = 9,
) -> np.ndarray:
    """
    A_rep / A_loc: (n_feats, n_beats)
    返回特征向量 evecs (n_beats, n_beats)
    """
    R = librosa.segment.recurrence_matrix(
        A_rep, width=rec_width, mode="affinity", metric="cosine", sym=True
    )
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, rec_smooth))

    path_distance = np.sum(np.diff(A_loc, axis=1) ** 2, axis=0)
    sigma = float(np.median(path_distance)) + 1e-8
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    denom = np.sum((deg_path + deg_rec) ** 2) + 1e-8
    mu = float(deg_path.dot(deg_path + deg_rec) / denom)
    A = mu * Rf + (1.0 - mu) * R_path

    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    _evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(evec_smooth, 1))
    return evecs


def _cluster_laplacian(
    evecs: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 bound_idxs（含首尾）, segment_labels（每段一个）"""
    Cnorm = np.cumsum(evecs**2, axis=1) ** 0.5
    X = evecs[:, :k] / (Cnorm[:, k - 1 : k] + 1e-8)
    seg_ids = KMeans(n_clusters=k, n_init=30, max_iter=400, random_state=0).fit_predict(X)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_idxs = librosa.util.fix_frames(bound_beats, x_min=0)
    bound_segs = list(seg_ids[bound_idxs])
    bound_idxs = np.append(bound_idxs, len(Cnorm) - 1)
    return np.asarray(bound_idxs, dtype=int), np.asarray(bound_segs, dtype=int)


def _estimate_k(n_beats: int, duration: float, user_k: int | None) -> int:
    if user_k is not None and user_k >= 2:
        return int(np.clip(user_k, 2, 10))
    # 流行曲常见 3–6 类结构单元；按时长估计
    k = int(round(duration / 45.0)) + 2
    return int(np.clip(k, 3, 7))


def _name_sections(
    bound_times: np.ndarray,
    labels: np.ndarray,
    energy_per_seg: np.ndarray,
    duration: float,
) -> list[str]:
    """
    基于重复次数 + 能量 + 位置命名。
    - 高能量段（接近峰值）→ Chorus
    - 重复出现的中等能量类 → Verse
    - 单次出现、中段、能量明显低于副歌 → Bridge
    """
    n = len(labels)
    names = [f"段落 {i + 1}" for i in range(n)]
    if n == 0:
        return names

    uniq = sorted(set(int(x) for x in labels))
    counts = {u: int(np.sum(labels == u)) for u in uniq}
    mean_e = {
        u: float(np.mean(energy_per_seg[labels == u])) if np.any(labels == u) else 0.0
        for u in uniq
    }
    max_e = float(np.max(energy_per_seg)) if len(energy_per_seg) else 0.0
    chorus_thr = max(0.55, max_e * 0.82)

    repeated = [u for u in uniq if counts[u] >= 2]
    if repeated:
        chorus_lab = max(repeated, key=lambda u: mean_e[u])
        verse_cands = [u for u in repeated if u != chorus_lab]
        verse_lab = max(verse_cands, key=lambda u: counts[u] * 10 + mean_e[u]) if verse_cands else None
    else:
        chorus_lab = max(uniq, key=lambda u: mean_e[u]) if uniq else 0
        verse_lab = None

    for i in range(n):
        lab = int(labels[i])
        e = float(energy_per_seg[i])
        start, end = float(bound_times[i]), float(bound_times[i + 1])
        mid = 0.5 * (start + end)
        pos = mid / max(duration, 1e-6)

        if i == 0 and e < 0.42 and end < duration * 0.28:
            names[i] = "前奏 / Intro"
        elif i == n - 1 and e < 0.50 and start > duration * 0.72:
            names[i] = "尾声 / Outro"
        elif e >= chorus_thr:
            names[i] = "副歌 / Chorus"
        elif lab == chorus_lab and e >= 0.48:
            names[i] = "副歌 / Chorus"
        elif verse_lab is not None and lab == verse_lab:
            names[i] = "主歌 / Verse"
        elif counts.get(lab, 0) >= 2:
            names[i] = "主歌 / Verse"
        elif counts.get(lab, 0) == 1 and 0.30 < pos < 0.85 and e <= chorus_thr * 0.72:
            names[i] = "桥段 / Bridge"
        else:
            names[i] = "过渡 / Transition"

    if not any("Chorus" in x for x in names):
        names[int(np.argmax(energy_per_seg))] = "副歌 / Chorus"
    return names


def analyze_structure(
    y: np.ndarray,
    sr: int,
    method: str = "scluster",
    n_sections: int | None = None,
    hop_length: int = 512,
) -> StructureResult:
    """
    method:
      - scluster: McFee Laplacian 谱聚类（推荐，ISMIR 2014）
      - foote:    Foote 新奇度峰值（ICME 2000）
      - hybrid:   Foote 边界 + Laplacian 标签
    n_sections: 目标段落数（用于估计聚类数 / 峰值数）；None=自动
    """
    duration = float(len(y) / sr)

    # --- beat 同步特征 ---
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if len(beats) < 8:
        # 回退：均匀假拍
        n_fake = max(16, int(duration / 0.5))
        beats = np.linspace(0, max(1, librosa.samples_to_frames(len(y), hop_length=hop_length) - 1), n_fake).astype(int)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    # pad=True → 列数 = len(beats)+1，对应区间 [0,b0), [b0,b1), ..., [b_last, end)
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median, pad=True)
    mfcc_sync = librosa.util.sync(mfcc, beats, aggregate=np.mean, pad=True)
    seg_frames = np.concatenate([[0], np.asarray(beats, dtype=int)])
    beat_times = librosa.frames_to_time(seg_frames, sr=sr, hop_length=hop_length)
    # 与 sync 列数对齐
    n_seg = chroma_sync.shape[1]
    if len(beat_times) > n_seg:
        beat_times = beat_times[:n_seg]
    elif len(beat_times) < n_seg:
        beat_times = np.append(beat_times, [duration] * (n_seg - len(beat_times)))
    beat_times = beat_times.astype(float)
    beat_times[0] = 0.0

    # stack for foote / 标签: (n_beats, feats)
    X_beat = np.vstack([chroma_sync, mfcc_sync]).T
    X_beat = (X_beat - X_beat.mean(axis=0)) / (X_beat.std(axis=0) + 1e-8)

    novelty_beat = foote_novelty(X_beat, M_gaussian=64, m_median=4)

    method = (method or "scluster").lower()
    k = _estimate_k(len(beat_times), duration, n_sections)

    if method == "foote":
        min_dist = max(3, int(8.0 / max(duration / max(len(beat_times), 1), 0.4)))  # ~8s
        # 约 8 秒最小段长 → beat 间距
        bt = np.diff(beat_times).mean() if len(beat_times) > 1 else 0.5
        min_peak_dist = max(2, int(8.0 / max(bt, 0.3)))
        peaks = foote_peaks(novelty_beat, L=16, min_peak_dist=min_peak_dist)
        if n_sections is not None and len(peaks) > n_sections - 1:
            # 保留最高峰
            scores = novelty_beat[peaks]
            peaks = np.sort(peaks[np.argsort(scores)[::-1][: n_sections - 1]])
        bound_beats = np.unique(np.concatenate([[0], peaks, [len(beat_times) - 1]])).astype(int)
        # 标签：用相邻段特征聚类
        seg_feats = []
        for a, b in zip(bound_beats[:-1], bound_beats[1:]):
            b = max(a + 1, b)
            seg_feats.append(X_beat[a:b].mean(axis=0))
        n_lab = min(k, len(seg_feats))
        if n_lab >= 2:
            labels = KMeans(n_clusters=n_lab, n_init=20, random_state=0).fit_predict(np.array(seg_feats))
        else:
            labels = np.zeros(len(seg_feats), dtype=int)
        used = "foote"

    else:
        # scluster / hybrid
        evecs = _embed_laplacian(chroma_sync, mfcc_sync)
        # 若用户指定段数，优先用 k≈段数；否则自动
        if n_sections is not None:
            k = int(np.clip(n_sections, 2, 10))
        bound_beats, labels = _cluster_laplacian(evecs, k)

        if method == "hybrid":
            # 用 Foote 峰微调：靠近谱聚类边界的峰合并，远处峰插入
            peaks = foote_peaks(novelty_beat, L=16, min_peak_dist=max(2, len(beat_times) // 30))
            merged = set(int(x) for x in bound_beats[:-1])
            for p in peaks:
                if any(abs(p - b) <= 2 for b in merged):
                    continue
                # 只插入较强峰
                if novelty_beat[p] >= 0.35:
                    merged.add(int(p))
            bound_beats = np.array(sorted(merged) + [len(beat_times) - 1], dtype=int)
            # 重新按边界取多数标签
            # 先在原 beat 上还原逐拍标签
            beat_lab = np.zeros(len(beat_times), dtype=int)
            for i, (a, b) in enumerate(zip(bound_beats[:-1], bound_beats[1:])):
                # 临时：用特征再聚类
                pass
            seg_feats = []
            for a, b in zip(bound_beats[:-1], bound_beats[1:]):
                b = max(a + 1, b)
                seg_feats.append(X_beat[a:b].mean(axis=0))
            n_lab = min(k, len(seg_feats))
            labels = KMeans(n_clusters=max(2, n_lab), n_init=20, random_state=0).fit_predict(
                np.array(seg_feats)
            )
            used = "hybrid"
        else:
            used = "scluster"

    bound_beats = np.clip(np.asarray(bound_beats, dtype=int), 0, len(beat_times) - 1)
    bound_beats = np.unique(bound_beats)
    if bound_beats[0] != 0:
        bound_beats = np.concatenate([[0], bound_beats])
    if bound_beats[-1] != len(beat_times) - 1:
        bound_beats = np.concatenate([bound_beats, [len(beat_times) - 1]])
    bound_times = beat_times[bound_beats].astype(float)
    bound_times[0] = 0.0
    bound_times[-1] = duration

    # 边界变更后按段特征重算标签，保证长度一致
    seg_feats = []
    for a, b in zip(bound_beats[:-1], bound_beats[1:]):
        b = max(a + 1, int(b))
        seg_feats.append(X_beat[a:b].mean(axis=0))
    seg_feats_a = np.asarray(seg_feats)
    n_lab = min(max(2, k), len(seg_feats_a))
    if len(seg_feats_a) >= 2:
        labels = KMeans(n_clusters=n_lab, n_init=20, random_state=0).fit_predict(seg_feats_a)
    else:
        labels = np.zeros(len(seg_feats_a), dtype=int)

    # 段能量（用 beat 同步 rms）
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_sync = librosa.util.sync(rms, beats, aggregate=np.mean, pad=True)
    rms_sync = np.asarray(rms_sync[: len(beat_times)], dtype=float)
    if len(rms_sync) < len(beat_times):
        rms_sync = np.pad(rms_sync, (0, len(beat_times) - len(rms_sync)), mode="edge")
    rms_n = (rms_sync - rms_sync.min()) / (rms_sync.max() - rms_sync.min() + 1e-8)

    energy_per_seg = []
    for a, b in zip(bound_beats[:-1], bound_beats[1:]):
        b = max(a + 1, b)
        energy_per_seg.append(float(np.mean(rms_n[a:b])))
    energy_per_seg = np.asarray(energy_per_seg)

    # 合并过短段（< 4 秒）
    min_dur = 4.0
    bt = list(bound_times)
    lb = list(labels)
    en = list(energy_per_seg)
    i = 0
    while i < len(lb):
        dur = bt[i + 1] - bt[i]
        if dur < min_dur and len(lb) > 2:
            # 合并到能量更接近的一侧
            if i == 0:
                # 并入右侧
                en[1] = (en[0] * dur + en[1] * (bt[2] - bt[1])) / (bt[2] - bt[0] + 1e-8)
                bt.pop(i + 1)
                lb.pop(i)
                en.pop(i)
            elif i == len(lb) - 1:
                en[i - 1] = (en[i - 1] * (bt[i] - bt[i - 1]) + en[i] * dur) / (bt[i + 1] - bt[i - 1] + 1e-8)
                bt.pop(i)
                lb.pop(i)
                en.pop(i)
            else:
                left_sim = abs(en[i] - en[i - 1])
                right_sim = abs(en[i] - en[i + 1])
                if left_sim <= right_sim:
                    en[i - 1] = (en[i - 1] * (bt[i] - bt[i - 1]) + en[i] * dur) / (bt[i + 1] - bt[i - 1] + 1e-8)
                    bt.pop(i)
                    lb.pop(i)
                    en.pop(i)
                    i -= 1
                else:
                    en[i + 1] = (en[i] * dur + en[i + 1] * (bt[i + 2] - bt[i + 1])) / (bt[i + 2] - bt[i] + 1e-8)
                    bt.pop(i + 1)
                    lb.pop(i)
                    en.pop(i)
                    i -= 1
        i += 1
    bound_times = np.asarray(bt, dtype=float)
    labels = np.asarray(lb, dtype=int)
    energy_per_seg = np.asarray(en, dtype=float)

    names = _name_sections(bound_times, labels, energy_per_seg, duration)

    # 新奇度映射到均匀时间轴（展示）
    n_show = 1000
    show_t = np.linspace(0, duration, n_show)
    novelty_show = np.interp(show_t, beat_times[: len(novelty_beat)], novelty_beat)

    # bound_frames 用分析 hop 近似
    bound_frames = librosa.time_to_frames(bound_times, sr=sr, hop_length=hop_length)

    return StructureResult(
        bound_frames=bound_frames,
        bound_times=bound_times,
        segment_labels=labels,
        section_names=names,
        novelty=novelty_show,
        novelty_times=show_t,
        method=used,
        n_clusters=int(len(set(int(x) for x in labels))),
    )
