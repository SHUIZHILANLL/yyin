"""节拍闪烁条：绿=低强度 / 红=高强度；亮度一致；半拍防倍速。"""

from __future__ import annotations

import json

import numpy as np
import streamlit.components.v1 as components

from analyzer import AnalysisResult


CHANNEL = "yyin-beat-v1"


def _visual_beats(beats: np.ndarray, tempo: float) -> list[float]:
    """视觉拍：中位 BPM>100 时隔拍取一，避免倍速闪太快。"""
    arr = np.asarray(beats, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return [round(float(t), 3) for t in arr]
    med = float(np.median(np.diff(arr)))
    bpm = 60.0 / max(med, 1e-6)
    if bpm > 100 or (tempo > 0 and bpm > tempo * 1.25):
        arr = arr[0::2]
        if len(arr) >= 2:
            med = float(np.median(np.diff(arr)))
    return [round(float(t), 3) for t in arr]


def render_beat_pulse(result: AnalysisResult, height: int = 86) -> None:
    tempo = float(getattr(result, "tempo", 120.0) or 120.0)
    beats = getattr(result, "beat_times", None)
    if beats is None or len(beats) == 0:
        dur = float(getattr(result, "duration", 60.0) or 60.0)
        step = 60.0 / max(tempo, 1.0)
        beats = np.arange(0.0, dur, step)
    beat_list = _visual_beats(np.asarray(beats, dtype=float), tempo)
    payload = {"tempo": tempo, "channel": CHANNEL, "beats": beat_list}
    data = json.dumps(payload, ensure_ascii=False)

    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  .stage {{
    margin: 0; padding: 8px 12px;
    border-radius: 10px;
    border: 2px solid #3a3a3a;
    background: #0e0e0e;
    color: #ece7e0;
    font-family: "DM Sans", "Segoe UI", sans-serif;
  }}
  .head {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 11px; color: #c9c1b6; margin-bottom: 6px;
    position: relative; z-index: 1;
  }}
  .tag {{
    display: inline-block; padding: 1px 7px; border-radius: 999px;
    background: #8a3a2a; color: #fff; font-weight: 700; font-size: 10px;
    margin-right: 6px;
  }}
  .head b {{ color: #fff; }}
  .strip {{
    height: 42px; border-radius: 8px;
    background: #050505;
    border: 1px solid #222;
    position: relative; overflow: hidden;
  }}
  .flash {{
    position: absolute; inset: 0;
    opacity: 0;
    will-change: opacity, background;
  }}
  .swatch {{
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    vertical-align: -1px; margin: 0 3px;
  }}
  .note {{ text-align: center; font-size: 10px; color: #8f877c; margin-top: 4px; }}
</style></head>
<body>
<div class="stage" id="stage">
  <div class="head">
    <span><span class="tag">BEAT</span>
      <span class="swatch" style="background:#00e676"></span>低
      <span class="swatch" style="background:#ff1744"></span>高 · 亮度一致
    </span>
    <span>局部 <b id="bpm">—</b> BPM · I=<b id="iv">0.00</b></span>
  </div>
  <div class="strip"><div class="flash" id="flash"></div></div>
  <div class="note">跟拍闪光 · 与上方播放同步</div>
</div>
<script>
const CFG = {data};
const BEATS = (CFG.beats || []).map(Number).filter((x) => isFinite(x)).sort((a,b)=>a-b);
const bpmEl = document.getElementById("bpm");
const ivEl = document.getElementById("iv");
const stage = document.getElementById("stage");
const flash = document.getElementById("flash");
bpmEl.textContent = Number(CFG.tempo || 120).toFixed(1);

let st = {{ t: 0, playing: false, intensity: 0.4, tempo: CFG.tempo || 120 }};
let smoothI = 0.4;
try {{
  const bus = new BroadcastChannel(CFG.channel);
  bus.onmessage = (ev) => {{ if (ev && ev.data) st = Object.assign(st, ev.data); }};
}} catch (e) {{}}

function mixRgb(i) {{
  // 高饱和：纯绿 → 纯红（中间过亮黄/橙），色差拉满
  const a = Math.max(0, Math.min(1, i));
  let r, g, b;
  if (a < 0.5) {{
    const u = a / 0.5;
    r = Math.round(0 + 255 * u);
    g = Math.round(230);
    b = Math.round(40 * (1 - u));
  }} else {{
    const u = (a - 0.5) / 0.5;
    r = 255;
    g = Math.round(230 * (1 - u));
    b = Math.round(20 + 40 * u);
  }}
  return [r, g, b];
}}

function lastBeatAge(t) {{
  const fallbackIbi = 60 / Math.max(40, Math.min(160, (st.tempo || CFG.tempo || 120) * 0.5));
  if (!BEATS.length) {{
    const ibi = fallbackIbi;
    const age = ((t % ibi) + ibi) % ibi;
    return {{ age, ibi }};
  }}
  let lo = 0, hi = BEATS.length - 1, idx = -1;
  while (lo <= hi) {{
    const mid = (lo + hi) >> 1;
    if (BEATS[mid] <= t) {{ idx = mid; lo = mid + 1; }}
    else hi = mid - 1;
  }}
  if (idx < 0) return {{ age: t, ibi: Math.max(0.45, BEATS[0] || fallbackIbi) }};
  const t0 = BEATS[idx];
  let ibi = (idx + 1 < BEATS.length) ? (BEATS[idx + 1] - t0)
           : (idx > 0 ? (t0 - BEATS[idx - 1]) : fallbackIbi);
  ibi = Math.max(0.45, Math.min(2.2, ibi || fallbackIbi));
  return {{ age: Math.max(0, t - t0), ibi }};
}}

// 按秒的稳定包络（不跟拍距比例抖）
function envelopeSec(age) {{
  const attack = 0.045;
  const hold = 0.090;
  const release = 0.280;
  if (age <= attack) {{
    const u = age / attack;
    return u * u * (3 - 2 * u);
  }}
  if (age <= attack + hold) return 1;
  const r = (age - attack - hold) / release;
  if (r >= 1) return 0;
  const fall = 1 - r;
  return fall * fall;
}}

let disp = 0;
const FLASH_PEAK = 1.0;
function paint() {{
  const intensity = Math.max(0, Math.min(1, st.intensity ?? 0.4));
  // 颜色慢跟，避免闪色抖动
  smoothI += (intensity - smoothI) * 0.08;
  const playing = !!st.playing;
  const {{ age, ibi }} = lastBeatAge(st.t || 0);
  const target = playing ? envelopeSec(age) * FLASH_PEAK : 0;
  // 升起快、落下稳
  const follow = target > disp ? 0.55 : 0.18;
  disp += (target - disp) * follow;

  ivEl.textContent = intensity.toFixed(2);
  bpmEl.textContent = (60 / ibi).toFixed(1);

  const [r, g, b] = mixRgb(smoothI);
  const a = Math.max(0, Math.min(1, disp));
  // 不透明实色铺满，颜色一眼能辨
  flash.style.background = "rgb(" + r + "," + g + "," + b + ")";
  flash.style.opacity = a.toFixed(3);
  stage.style.borderColor = a > 0.05
    ? ("rgba(" + r + "," + g + "," + b + "," + (0.55 + 0.45 * a).toFixed(2) + ")")
    : "#3a3a3a";
  stage.style.boxShadow = a > 0.05
    ? ("0 0 " + Math.round(14 + a * 32) + "px rgba(" + r + "," + g + "," + b + "," +
       (0.35 + 0.5 * a).toFixed(2) + ")")
    : "none";
  requestAnimationFrame(paint);
}}
requestAnimationFrame(paint);
</script>
</body></html>"""
    components.html(html_doc, height=height, scrolling=False)
