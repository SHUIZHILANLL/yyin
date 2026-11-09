"""音乐段落 & 情绪强度分析 — Streamlit 界面（分阶段加载）。"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from analyzer import analyze_core, analyze_style_stage, format_time
from sync_player import render_sync_timeline
from visualize import fig_radar, fig_section_bars, fig_style_bars

st.set_page_config(
    page_title="音乐段落情绪分析",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Instrument Serif', serif !important; font-weight: 400 !important; }
.block-container { padding-top: 1.4rem; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _audio_bytes_for_range(path: str, start: float, end: float) -> bytes:
    data, sr = sf.read(path, always_2d=False)
    i0 = max(0, int(start * sr))
    i1 = min(len(data), int(end * sr))
    if i1 <= i0:
        i1 = min(len(data), i0 + int(0.5 * sr))
    clip = np.asarray(data[i0:i1])
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV")
    return buf.getvalue()


def _guess_mime(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }.get(ext, "audio/wav")


st.title("音乐段落 · 情绪强度分析")
st.caption("分阶段：结构/情绪 → 风格；情绪 = Russell VA + 风格先验 · 强度曲内对比拉伸")

with st.sidebar:
    st.header("输入")
    uploaded = st.file_uploader(
        "音乐文件",
        type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
    )
    n_sections = st.slider("目标段落数 / 聚类数（0 = 自动）", 0, 12, 0)
    method = st.selectbox(
        "结构划分算法",
        options=["scluster", "hybrid", "foote"],
        index=0,
        format_func=lambda m: {
            "scluster": "谱聚类 scluster（推荐）",
            "hybrid": "混合 hybrid",
            "foote": "Foote 新奇度",
        }[m],
    )
    sr = st.selectbox("分析采样率", [22050, 16000, 44100], index=0)
    auto_deep = st.checkbox("核心完成后自动分析风格", value=True)
    run = st.button("① 开始核心分析", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        """
**加载顺序**
1. 核心：分段 + 声学情绪（先出结果）
2. 风格：GTZAN → 校正情绪标签/强度

**情绪**
Russell 效价-唤醒 · 风格先验 · 强度对比拉伸
"""
    )

if run:
    if uploaded is None:
        st.warning("请先上传音乐文件。")
        st.stop()
    audio_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    with st.spinner("① 核心分析：结构分段 + 情绪曲线…"):
        result = analyze_core(
            tmp_path,
            sr=int(sr),
            n_sections=None if n_sections == 0 else int(n_sections),
            method=method,
        )
    st.session_state["result"] = result
    st.session_state["filename"] = uploaded.name
    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["audio_path"] = tmp_path
    st.session_state["audio_mime"] = _guess_mime(uploaded.name)
    st.session_state["pending_style"] = True
    st.session_state["auto_deep"] = auto_deep
    st.rerun()

if "result" not in st.session_state:
    st.info("在左侧上传音乐并点击「① 开始核心分析」。")
    st.stop()

result = st.session_state["result"]
filename = st.session_state.get("filename", Path(result.path).name)
audio_bytes = st.session_state.get("audio_bytes")
audio_path = st.session_state.get("audio_path", result.path)
audio_mime = st.session_state.get("audio_mime", "audio/wav")
auto_deep = st.session_state.get("auto_deep", True)

b1, b2 = st.columns([1, 3])
with b1:
    do_style = st.button(
        "② 分析风格",
        use_container_width=True,
        disabled=result.style is not None,
    )
with b2:
    st.caption(
        "进度："
        + " → ".join(["核心✓", "风格✓" if result.style else "风格…"])
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("文件", filename)
c2.metric("时长", format_time(result.duration))
c3.metric("估算 BPM", f"{result.tempo:.1f}")
c4.metric("段落数", len(result.sections))
c5.metric("算法", result.method)

m1, m2 = st.columns(2)
m1.metric(
    "风格",
    f"{result.style.top_style_zh} / {result.style.top_style}" if result.style else "待分析",
)
m2.metric("风格置信", f"{result.style.confidence:.0%}" if result.style else "—")
if result.style is not None:
    st.caption("风格标签：" + " · ".join(result.style.tags))

st.subheader("原曲播放 · 曲式时间线 / 全曲曲线同步")
render_sync_timeline(result, audio_bytes, mime=audio_mime, height=980)

left, right = st.columns([1.4, 1])
with left:
    st.subheader("各段落指数对比")
    st.plotly_chart(fig_section_bars(result), use_container_width=True)
with right:
    st.subheader("段落雷达 + 试听")
    idx = st.selectbox(
        "选择段落",
        options=list(range(len(result.sections))),
        format_func=lambda i: (
            f"#{result.sections[i].index} {result.sections[i].label} · "
            f"{result.sections[i].emotion_tag} · I={result.sections[i].intensity:.2f}"
        ),
    )
    sec = result.sections[idx]
    st.markdown(
        f"**#{sec.index} {sec.label}**　"
        f"`{format_time(sec.start)}`–`{format_time(sec.end)}`　"
        f"情绪：**{sec.emotion_tag}**　A={sec.arousal:.2f} V={sec.valence:.2f} I={sec.intensity:.2f}"
    )
    try:
        st.audio(_audio_bytes_for_range(audio_path, sec.start, sec.end), format="audio/wav")
    except Exception as exc:
        st.caption(str(exc))
    st.plotly_chart(fig_radar(result, idx), use_container_width=True)

st.subheader("段落明细")
df = pd.DataFrame(
    [
        {
            "序号": s.index,
            "标签": s.label,
            "开始": format_time(s.start),
            "结束": format_time(s.end),
            "情绪标签": s.emotion_tag,
            "能量": s.energy,
            "唤醒度": s.arousal,
            "效价": s.valence,
            "情绪强度": s.intensity,
            "局部BPM": s.tempo_local,
        }
        for s in result.sections
    ]
)
st.dataframe(df, use_container_width=True, hide_index=True)

run_s = do_style or (
    auto_deep and st.session_state.get("pending_style", False) and result.style is None
)
if run_s:
    with st.spinner("② 分析风格，并校正情绪…"):
        result = analyze_style_stage(result)
        result._y = None
        st.session_state["result"] = result
        st.session_state["pending_style"] = False
    st.rerun()

if result.style is not None:
    st.subheader("音乐风格（GTZAN）")
    st.plotly_chart(fig_style_bars(result.style), use_container_width=True)
else:
    st.info("风格待加载（自动进行中，或点「② 分析风格」）。")

st.download_button(
    "导出段落 CSV",
    df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{Path(filename).stem}_sections.csv",
    mime="text/csv",
)
