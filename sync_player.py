"""原曲 + 曲式时间线 + 全曲曲线 同步播放头（HTML/JS 组件）。"""

from __future__ import annotations

import base64
import html
import json
from typing import Any

import numpy as np
import streamlit.components.v1 as components

from analyzer import AnalysisResult
from visualize import SECTION_COLORS


def _fmt_js(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


def _downsample(x: np.ndarray, y: np.ndarray, n: int = 480) -> tuple[list[float], list[float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= n:
        return x.round(3).tolist(), y.round(4).tolist()
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx].round(3).tolist(), y[idx].round(4).tolist()


def render_sync_timeline(
    result: AnalysisResult,
    audio_bytes: bytes | None,
    mime: str = "audio/wav",
    height: int = 980,
) -> None:
    """渲染带同步时间标的曲式时间线 + 全曲曲线。"""
    # 只用结构字段，避免风格校正后 emotion/intensity 变化导致播放器 HTML 重挂载打断播放
    sections: list[dict[str, Any]] = []
    for i, sec in enumerate(result.sections):
        sections.append(
            {
                "index": sec.index,
                "label": sec.label,
                "start": sec.start,
                "end": sec.end,
                "emotion": "",
                "intensity": 0.0,
                "color": SECTION_COLORS[i % len(SECTION_COLORS)],
            }
        )

    duration = float(result.duration)
    step = 10.0 if duration <= 120 else (15.0 if duration <= 300 else 30.0)
    ticks = []
    t = 0.0
    while t <= duration + 1e-6:
        ticks.append({"t": round(t, 2), "label": _fmt_js(t)})
        t += step
    if not ticks or abs(ticks[-1]["t"] - duration) > 0.5:
        ticks.append({"t": round(duration, 2), "label": _fmt_js(duration)})

    wave_t = np.linspace(0, duration, len(result.waveform))
    wt, wy = _downsample(wave_t, result.waveform, 600)
    et, ey = _downsample(result.times, result.energy_curve, 480)
    _, ny = _downsample(result.novelty_times, result.novelty, 480)
    _, ay = _downsample(result.times, result.arousal_curve, 480)
    _, vy = _downsample(result.times, result.valence_curve, 480)
    _, iy = _downsample(result.times, result.intensity_curve, 480)

    audio_src = ""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_src = f"data:{mime};base64,{b64}"

    tempo = float(getattr(result, "tempo", 120.0) or 120.0)
    raw_beats = getattr(result, "beat_times", None)
    # 视觉闪光：用检测拍锁定相位，再铺均匀格子（约 66 BPM），避免疏密不均导致忽闪忽不闪
    target_ibi = 60.0 / 66.0
    phase0 = 0.0
    if raw_beats is not None and len(raw_beats) >= 2:
        beat_arr = np.asarray(raw_beats, dtype=float)
        beat_arr = beat_arr[np.isfinite(beat_arr)]
        if len(beat_arr) >= 2:
            med = float(np.median(np.diff(beat_arr)))
            # 折到约 55–75 BPM 的拍距
            while med > 0 and (60.0 / med) > 78 and med < 2.0:
                med *= 2.0
            while med > 0 and (60.0 / med) < 52 and med > 0.2:
                med *= 0.5
            target_ibi = float(np.clip(med, 60.0 / 78.0, 60.0 / 52.0))
            phase0 = float(beat_arr[0] % target_ibi)
    beat_list = [
        round(float(t), 3)
        for t in np.arange(phase0, duration + 1e-6, target_ibi)
        if 0.0 <= t <= duration
    ]

    payload = {
        "duration": duration,
        "tempo": tempo,
        "beats": beat_list,
        "sections": sections,
        "ticks": ticks,
        "audioSrc": audio_src,
        "wave": {"t": wt, "y": wy},
        "energy": {"t": et, "y": ey},
        "novelty": {"t": et, "y": ny},
        "arousal": {"t": et, "y": ay},
        "valence": {"t": et, "y": vy},
        "intensity": {"t": et, "y": iy},
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --ink: #1c1a17;
    --muted: #6b6560;
    --line: #d8d2c8;
    --play: #c23b22;
    --panel: #fbfaf7;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2px 0 0;
    font-family: "DM Sans", "Segoe UI", sans-serif;
    color: var(--ink); background: transparent;
  }}
  .wrap {{
    background: linear-gradient(180deg, #fbfaf7 0%, #f3efe8 100%);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 8px 12px 6px;
  }}
  .row {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
  }}
  .clock {{
    font-variant-numeric: tabular-nums;
    font-size: 20px; font-weight: 700; letter-spacing: 0.02em;
    min-width: 156px;
  }}
  .clock span.sep {{ color: var(--muted); font-weight: 500; margin: 0 4px; }}
  .meta {{ flex: 1; font-size: 13px; color: var(--muted); line-height: 1.35; }}
  .meta b {{ color: var(--ink); font-weight: 600; }}
  audio {{ width: 100%; height: 32px; margin-bottom: 6px; }}
  .rail-wrap {{ position: relative; padding-top: 2px; user-select: none; }}
  .rail {{
    position: relative; height: 36px; border-radius: 8px;
    background: #e8e2d8; overflow: hidden; cursor: pointer;
  }}
  .seg {{
    position: absolute; top: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    border-right: 1px solid rgba(255,255,255,.45);
    font-size: 11px; font-weight: 600; color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,.35);
    overflow: hidden; white-space: nowrap; padding: 0 4px;
  }}
  .playhead {{
    position: absolute; top: -3px; bottom: -3px; width: 2px;
    background: var(--play); left: 0%;
    box-shadow: 0 0 0 1px rgba(255,255,255,.7);
    pointer-events: none; z-index: 5;
  }}
  .playhead::before {{
    content: "";
    position: absolute; top: -2px; left: -5px;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid var(--play);
  }}
  .ticks {{ position: relative; height: 24px; margin-top: 2px; }}
  .tick {{
    position: absolute; transform: translateX(-50%);
    font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .tick::before {{
    content: ""; display: block; width: 1px; height: 6px;
    background: #b8b0a4; margin: 0 auto 2px;
  }}
  .charts {{ margin-top: 4px; display: flex; flex-direction: column; gap: 4px; }}
  .panel {{
    position: relative; background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px;
    overflow: hidden; cursor: pointer;
  }}
  .panel-title {{
    position: absolute; left: 8px; top: 3px; z-index: 3;
    font-size: 10px; color: var(--muted); background: rgba(251,250,247,.85);
    padding: 1px 6px; border-radius: 4px; pointer-events: none;
  }}
  .panel canvas {{ display: block; width: 100%; height: 72px; }}
  .cplay {{
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: var(--play); left: 0; pointer-events: none; z-index: 4;
  }}
  .legend {{
    display: flex; flex-wrap: wrap; gap: 10px 14px;
    font-size: 11px; color: var(--muted); margin: 4px 0 2px; padding-left: 2px;
  }}
  .legend i {{
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 4px; vertical-align: -1px;
  }}
  .hint {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  .beat-box {{
    margin-top: 8px; padding: 8px 10px; border-radius: 10px;
    border: 2px solid #3a3a3a; background: #0e0e0e; color: #eee;
  }}
  .beat-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 11px; color: #c9c1b6; margin-bottom: 6px;
  }}
  .beat-head .tag {{
    display: inline-block; padding: 1px 7px; border-radius: 999px;
    background: #8a3a2a; color: #fff; font-weight: 700; font-size: 10px; margin-right: 6px;
  }}
  .beat-head b {{ color: #fff; }}
  .swatch {{
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    vertical-align: -1px; margin: 0 3px;
  }}
  .beat-strip {{
    height: 40px; border-radius: 8px; background: #050505; border: 1px solid #222;
    position: relative; overflow: hidden;
  }}
  .beat-flash {{ position: absolute; inset: 0; opacity: 0; background: #00e676; }}
  .beat-note {{ text-align: center; font-size: 10px; color: #8f877c; margin-top: 4px; }}
</style>
</head>
<body>
<!-- beat-inline-v7-stable-pulse -->
<div class="wrap">
  <div class="beat-box" id="beatBox">
    <div class="beat-head">
      <span><span class="tag">BEAT</span>
        <span class="swatch" style="background:#00e676"></span>低
        <span class="swatch" style="background:#ff1744"></span>高 · 约60–75BPM
      </span>
      <span>局部 <b id="beatBpm">—</b> BPM · I=<b id="beatI">0.00</b></span>
    </div>
    <div class="beat-strip"><div class="beat-flash" id="beatFlash"></div></div>
    <div class="beat-note">绿=情绪低 · 红=情绪高 · 点下方播放键后闪烁</div>
  </div>
  <div class="row">
    <div class="clock" id="clock">00:00.0<span class="sep">/</span>00:00.0</div>
    <div class="meta" id="meta">当前段落：—</div>
  </div>
  <audio id="player" controls preload="auto" src="{html.escape(audio_src)}"></audio>
  <div class="rail-wrap">
    <div class="rail" id="rail"></div>
    <div class="playhead" id="playhead"></div>
    <div class="ticks" id="ticks"></div>
  </div>
  <div class="legend">
    <span><i style="background:#3D5A80"></i>波形</span>
    <span><i style="background:#F58518"></i>能量</span>
    <span><i style="background:#9D755D"></i>结构新奇度</span>
    <span><i style="background:#E45756"></i>唤醒度</span>
    <span><i style="background:#54A24B"></i>效价</span>
    <span><i style="background:#4C78A8"></i>情绪强度</span>
    <span><i style="background:#c23b22"></i>时间标</span>
  </div>
  <div class="charts" id="charts"></div>
  <div class="hint">点击色块或曲线区域可跳转 · 红线为当前播放位置</div>
</div>
<script>
const DATA = {data_json};

function fmt(sec) {{
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return String(m).padStart(2, "0") + ":" + s.toFixed(1).padStart(4, "0");
}}

const rail = document.getElementById("rail");
const playhead = document.getElementById("playhead");
const ticksEl = document.getElementById("ticks");
const clock = document.getElementById("clock");
const meta = document.getElementById("meta");
const player = document.getElementById("player");
const chartsEl = document.getElementById("charts");
const duration = DATA.duration || 1;
const tempo = Math.max(40, Math.min(220, DATA.tempo || 120));
const BEATS = (DATA.beats || []).map(Number).filter((x) => isFinite(x)).sort((a,b)=>a-b);
const curveHeads = [];
const beatFlash = document.getElementById("beatFlash");
const beatBox = document.getElementById("beatBox");
const beatBpm = document.getElementById("beatBpm");
const beatI = document.getElementById("beatI");
beatBpm.textContent = tempo.toFixed(1);

function sampleIntensity(t) {{
  const series = DATA.intensity;
  const ts = series.t, ys = series.y;
  if (!ts || !ts.length) return 0.45;
  if (t <= ts[0]) return ys[0];
  if (t >= ts[ts.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) {{
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid;
  }}
  const span = ts[hi] - ts[lo] || 1;
  const u = (t - ts[lo]) / span;
  return ys[lo] * (1 - u) + ys[hi] * u;
}}

function mixRgb(i) {{
  const a = Math.max(0, Math.min(1, i));
  let r, g, b;
  if (a < 0.5) {{
    const u = a / 0.5;
    r = Math.round(255 * u); g = 230; b = Math.round(40 * (1 - u));
  }} else {{
    const u = (a - 0.5) / 0.5;
    r = 255; g = Math.round(230 * (1 - u)); b = Math.round(20 + 40 * u);
  }}
  return [r, g, b];
}}

function findBeatIndex(t) {{
  if (!BEATS.length) return -1;
  let lo = 0, hi = BEATS.length - 1, idx = -1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (BEATS[mid] <= t + 0.02) {{ idx = mid; lo = mid + 1; }} else hi = mid - 1;
  }}
  return idx;
}}

const VISUAL_IBI = (BEATS.length >= 2)
  ? Math.max(0.75, Math.min(1.2, BEATS[1] - BEATS[0]))
  : (60 / 66);
let lastBeatIdx = -1;
let lastAudioT = 0;
let pulse = 0;
let lastWall = performance.now();
let smoothI = 0.4;
const PULSE_DECAY_SEC = 0.38; // 每次拍固定亮这么久，不跟疏密抖动

function updateBeatFlash(t) {{
  const now = performance.now();
  const dt = Math.min(0.05, Math.max(0.001, (now - lastWall) / 1000));
  lastWall = now;
  const playing = !!(player && !player.paused && !player.ended);
  const intensity = sampleIntensity(t);
  smoothI += (intensity - smoothI) * 0.1;

  // 拖动进度条：重置索引，避免连触发或漏触发
  if (Math.abs(t - lastAudioT) > 0.4) {{
    lastBeatIdx = findBeatIndex(t);
  }}
  lastAudioT = t;

  const idx = findBeatIndex(t);
  if (playing && idx >= 0 && idx !== lastBeatIdx) {{
    pulse = 1.0; // 边沿硬触发：过拍必闪
    lastBeatIdx = idx;
  }}
  if (!playing) {{
    pulse = 0;
  }} else {{
    pulse = Math.max(0, pulse - dt / PULSE_DECAY_SEC);
  }}

  const a = pulse;
  const [r, g, b] = mixRgb(smoothI);
  beatFlash.style.background = "rgb(" + r + "," + g + "," + b + ")";
  beatFlash.style.opacity = a.toFixed(3);
  beatBpm.textContent = (60 / VISUAL_IBI).toFixed(1);
  beatI.textContent = intensity.toFixed(2);
  beatBox.style.borderColor = a > 0.05
    ? ("rgba(" + r + "," + g + "," + b + "," + (0.55 + 0.45 * a).toFixed(2) + ")")
    : "#3a3a3a";
  beatBox.style.boxShadow = a > 0.05
    ? ("0 0 " + Math.round(14 + a * 32) + "px rgba(" + r + "," + g + "," + b + "," +
       (0.35 + 0.5 * a).toFixed(2) + ")")
    : "none";
}}

DATA.sections.forEach((s) => {{
  const el = document.createElement("div");
  el.className = "seg";
  el.style.left = (s.start / duration) * 100 + "%";
  el.style.width = Math.max(0.4, ((s.end - s.start) / duration) * 100) + "%";
  el.style.background = s.color;
  el.textContent = "#" + s.index;
  el.title = "#" + s.index + " " + s.label + "\\n" +
    fmt(s.start) + " → " + fmt(s.end);
  el.addEventListener("click", (e) => {{
    e.stopPropagation();
    seekTo(s.start, true);
  }});
  rail.appendChild(el);
}});

DATA.ticks.forEach((tk) => {{
  const el = document.createElement("div");
  el.className = "tick";
  el.style.left = (tk.t / duration) * 100 + "%";
  el.textContent = tk.label;
  ticksEl.appendChild(el);
}});

function findSection(t) {{
  for (const s of DATA.sections) {{
    if (t >= s.start && t < s.end) return s;
  }}
  return DATA.sections[DATA.sections.length - 1] || null;
}}

function seekTo(t, play) {{
  player.currentTime = Math.min(duration, Math.max(0, t));
  if (play) player.play();
  sync(player.currentTime);
}}

function bindSeek(el) {{
  el.addEventListener("click", (e) => {{
    const rect = el.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    seekTo(ratio * duration, false);
  }});
}}
bindSeek(rail);

function makePanel(title, drawFn) {{
  const panel = document.createElement("div");
  panel.className = "panel";
  const label = document.createElement("div");
  label.className = "panel-title";
  label.textContent = title;
  const canvas = document.createElement("canvas");
  const head = document.createElement("div");
  head.className = "cplay";
  panel.appendChild(label);
  panel.appendChild(canvas);
  panel.appendChild(head);
  chartsEl.appendChild(panel);
  curveHeads.push(head);
  bindSeek(panel);
  return {{ panel, canvas, head, drawFn }};
}}

function setupCanvas(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
    const h = canvas.clientHeight || 72;
  canvas.width = Math.max(1, Math.floor(w * dpr));
  canvas.height = Math.max(1, Math.floor(h * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {{ ctx, w, h }};
}}

function xOf(t, w, pad) {{
  return pad + ((t / duration) * (w - pad * 2));
}}

function drawSectionBands(ctx, w, h, pad) {{
  DATA.sections.forEach((s) => {{
    const x0 = xOf(s.start, w, pad);
    const x1 = xOf(s.end, w, pad);
    ctx.fillStyle = s.color + "22";
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    ctx.strokeStyle = s.color + "66";
    ctx.beginPath();
    ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
  }});
}}

function drawSeries(ctx, w, h, pad, series, color, yMin, yMax, fill) {{
  const t = series.t, y = series.y;
  if (!t || !t.length) return;
  const span = (yMax - yMin) || 1;
  ctx.beginPath();
  for (let i = 0; i < t.length; i++) {{
    const px = xOf(t[i], w, pad);
    const py = h - pad - ((y[i] - yMin) / span) * (h - pad * 2);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.stroke();
  if (fill) {{
    const lastX = xOf(t[t.length - 1], w, pad);
    const base = h - pad;
    ctx.lineTo(lastX, base);
    ctx.lineTo(xOf(t[0], w, pad), base);
    ctx.closePath();
    ctx.fillStyle = color + "33";
    ctx.fill();
  }}
}}

function drawWave(ctx, w, h) {{
  const pad = 8;
  ctx.clearRect(0, 0, w, h);
  drawSectionBands(ctx, w, h, pad);
  const y = DATA.wave.y;
  let lo = -1, hi = 1;
  drawSeries(ctx, w, h, pad, DATA.wave, "#3D5A80", lo, hi, false);
}}

function drawEnergy(ctx, w, h) {{
  const pad = 8;
  ctx.clearRect(0, 0, w, h);
  drawSectionBands(ctx, w, h, pad);
  drawSeries(ctx, w, h, pad, DATA.energy, "#F58518", 0, 1, false);
  drawSeries(ctx, w, h, pad, DATA.novelty, "#9D755D", 0, 1, false);
}}

function drawMood(ctx, w, h) {{
  const pad = 8;
  ctx.clearRect(0, 0, w, h);
  drawSectionBands(ctx, w, h, pad);
  drawSeries(ctx, w, h, pad, DATA.arousal, "#E45756", 0, 1, false);
  drawSeries(ctx, w, h, pad, DATA.valence, "#54A24B", 0, 1, false);
}}

function drawIntensity(ctx, w, h) {{
  const pad = 8;
  ctx.clearRect(0, 0, w, h);
  drawSectionBands(ctx, w, h, pad);
  drawSeries(ctx, w, h, pad, DATA.intensity, "#4C78A8", 0, 1, true);
}}

const panels = [
  makePanel("波形", drawWave),
  makePanel("能量 / 结构新奇度", drawEnergy),
  makePanel("唤醒度 · 效价", drawMood),
  makePanel("情绪强度指数", drawIntensity),
];

function redrawAll() {{
  panels.forEach((p) => {{
    const {{ ctx, w, h }} = setupCanvas(p.canvas);
    p.drawFn(ctx, w, h);
  }});
}}

function sync(t) {{
  const pct = Math.min(100, Math.max(0, (t / duration) * 100));
  playhead.style.left = pct + "%";
  curveHeads.forEach((h) => {{ h.style.left = pct + "%"; }});
  clock.innerHTML = fmt(t) + '<span class="sep">/</span>' + fmt(duration);
  const s = findSection(t);
  if (s) {{
    meta.innerHTML = "当前段落：<b>#" + s.index + " " + s.label + "</b> · " +
      fmt(s.start) + "–" + fmt(s.end);
  }}
  updateBeatFlash(t);
}}

player.addEventListener("timeupdate", () => sync(player.currentTime || 0));
player.addEventListener("seeked", () => sync(player.currentTime || 0));
player.addEventListener("loadedmetadata", () => sync(player.currentTime || 0));
player.addEventListener("play", () => sync(player.currentTime || 0));
player.addEventListener("pause", () => sync(player.currentTime || 0));
window.addEventListener("resize", () => {{ redrawAll(); sync(player.currentTime || 0); }});

function loop() {{
  sync(player.currentTime || 0);
  requestAnimationFrame(loop);
}}
requestAnimationFrame(() => {{ redrawAll(); sync(0); loop(); }});
</script>
</body>
</html>
"""
    components.html(html_doc, height=max(height, 680), scrolling=True)
