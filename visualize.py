"""Plotly 可视化：波形、情绪曲线、段落总览、风格。"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer import AnalysisResult, format_time
from style import StyleResult


SECTION_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#1F9E89",
    "#D95F02",
]


def _section_shapes(result: AnalysisResult, y0: float, y1: float) -> list[dict]:
    shapes = []
    for i, sec in enumerate(result.sections):
        color = SECTION_COLORS[i % len(SECTION_COLORS)]
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=sec.start,
                x1=sec.end,
                y0=y0,
                y1=y1,
                fillcolor=color,
                opacity=0.18,
                line_width=0,
                layer="below",
            )
        )
    return shapes


def fig_overview(result: AnalysisResult) -> go.Figure:
    wave_t = (
        __import__("numpy").linspace(0, result.duration, len(result.waveform))
    )
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.22, 0.26, 0.26, 0.26],
        vertical_spacing=0.04,
        subplot_titles=("波形", "能量 / 结构新奇度", "唤醒度 · 效价", "情绪强度指数"),
    )

    fig.add_trace(
        go.Scatter(x=wave_t, y=result.waveform, name="波形", line=dict(width=0.6, color="#3D5A80"), hoverinfo="skip"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.times, y=result.energy_curve, name="能量", line=dict(color="#F58518", width=1.8)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.novelty_times, y=result.novelty, name="结构新奇度", line=dict(color="#9D755D", width=1.2, dash="dot")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.times, y=result.arousal_curve, name="唤醒度 Arousal", line=dict(color="#E45756", width=1.8)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=result.times, y=result.valence_curve, name="效价 Valence", line=dict(color="#54A24B", width=1.8)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.times,
            y=result.intensity_curve,
            name="情绪强度",
            fill="tozeroy",
            line=dict(color="#4C78A8", width=2),
        ),
        row=4,
        col=1,
    )

    for i, sec in enumerate(result.sections):
        color = SECTION_COLORS[i % len(SECTION_COLORS)]
        for r in range(1, 5):
            fig.add_vline(x=sec.start, line_width=1, line_dash="dash", line_color=color, opacity=0.7, row=r, col=1)
        fig.add_annotation(
            x=(sec.start + sec.end) / 2,
            y=1.05,
            text=f"{sec.index}",
            showarrow=False,
            xref="x",
            yref="y4 domain",
            font=dict(size=11, color=color),
        )

    fig.update_layout(
        height=820,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="时间 (秒)", row=4, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    for r in (2, 3, 4):
        fig.update_yaxes(range=[0, 1.05], row=r, col=1)
    return fig


def fig_section_bars(result: AnalysisResult) -> go.Figure:
    labels = [f"#{s.index} {s.label}" for s in result.sections]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="能量", x=labels, y=[s.energy for s in result.sections], marker_color="#F58518")
    )
    fig.add_trace(
        go.Bar(name="唤醒度", x=labels, y=[s.arousal for s in result.sections], marker_color="#E45756")
    )
    fig.add_trace(
        go.Bar(name="效价", x=labels, y=[s.valence for s in result.sections], marker_color="#54A24B")
    )
    fig.add_trace(
        go.Bar(name="情绪强度", x=labels, y=[s.intensity for s in result.sections], marker_color="#4C78A8")
    )
    fig.update_layout(
        barmode="group",
        height=420,
        template="plotly_white",
        yaxis=dict(range=[0, 1], title="指数 0–1"),
        margin=dict(l=40, r=20, t=30, b=80),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def fig_radar(result: AnalysisResult, section_index: int = 0) -> go.Figure:
    sec = result.sections[section_index]
    cats = ["能量", "唤醒度", "效价", "情绪强度", "明亮度", "频谱对比"]
    vals = [sec.energy, sec.arousal, sec.valence, sec.intensity, sec.brightness, sec.spectral_contrast]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=f"#{sec.index} {sec.label}",
            line_color=SECTION_COLORS[section_index % len(SECTION_COLORS)],
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        title=f"段落雷达 · {sec.emotion_tag} · {format_time(sec.start)}–{format_time(sec.end)}",
        template="plotly_white",
    )
    return fig


def fig_timeline(result: AnalysisResult) -> go.Figure:
    fig = go.Figure()
    for i, sec in enumerate(result.sections):
        color = SECTION_COLORS[i % len(SECTION_COLORS)]
        fig.add_trace(
            go.Bar(
                x=[sec.duration],
                y=["曲式结构"],
                base=[sec.start],
                orientation="h",
                name=f"#{sec.index} {sec.label}",
                marker_color=color,
                hovertemplate=(
                    f"<b>#{sec.index} {sec.label}</b><br>"
                    f"{format_time(sec.start)} → {format_time(sec.end)}<br>"
                    f"情绪: {sec.emotion_tag}<br>"
                    f"强度: {sec.intensity:.2f}<br>"
                    f"唤醒/效价: {sec.arousal:.2f} / {sec.valence:.2f}"
                    "<extra></extra>"
                ),
            )
        )
    # mm:ss 时间刻度
    step = 10 if result.duration <= 120 else (15 if result.duration <= 300 else 30)
    tickvals = list(range(0, int(result.duration) + 1, step))
    if not tickvals or tickvals[-1] < result.duration:
        tickvals.append(round(result.duration, 1))
    ticktext = [format_time(float(v)) for v in tickvals]
    fig.update_layout(
        barmode="stack",
        height=240,
        template="plotly_white",
        xaxis=dict(
            title="时间",
            range=[0, result.duration],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=50),
        showlegend=True,
        legend=dict(orientation="h", y=-0.55),
    )
    return fig


def fig_style_bars(style: StyleResult) -> go.Figure:
    labels = [f"{s.label_zh}" for s in style.scores[:10]]
    vals = [s.score for s in style.scores[:10]]
    fig = go.Figure(
        go.Bar(
            x=vals[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=["#E45756" if i == len(vals) - 1 else "#72B7B2" for i in range(len(vals))],
        )
    )
    fig.update_layout(
        height=360,
        template="plotly_white",
        xaxis=dict(title="置信度", range=[0, 1]),
        margin=dict(l=80, r=20, t=40, b=40),
        title=f"风格 Top：{style.top_style_zh} ({style.confidence:.0%}) · {style.backend}",
    )
    return fig
