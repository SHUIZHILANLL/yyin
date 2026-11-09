"""命令行：分析音乐并导出图表 / JSON / CSV。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyzer import analyze_music, format_time
from visualize import fig_overview, fig_section_bars, fig_timeline


def main() -> None:
    p = argparse.ArgumentParser(description="音乐段落与情绪强度分析")
    p.add_argument("audio", help="音频文件路径")
    p.add_argument("-o", "--out", default="output", help="输出目录")
    p.add_argument("-n", "--sections", type=int, default=0, help="目标段落数，0=自动")
    p.add_argument(
        "-m",
        "--method",
        default="scluster",
        choices=["scluster", "hybrid", "foote"],
        help="结构算法: scluster|hybrid|foote",
    )
    p.add_argument("--sr", type=int, default=22050, help="采样率")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem

    print(f"[1/3] 分析 {args.audio} [{args.method}] …")
    result = analyze_music(
        args.audio,
        sr=args.sr,
        n_sections=None if args.sections == 0 else args.sections,
        method=args.method,
    )

    summary = result.to_summary()
    json_path = out / f"{stem}_analysis.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {
            "index": s.index,
            "label": s.label,
            "start": s.start,
            "end": s.end,
            "start_fmt": format_time(s.start),
            "end_fmt": format_time(s.end),
            "duration": s.duration,
            "emotion_tag": s.emotion_tag,
            "energy": s.energy,
            "arousal": s.arousal,
            "valence": s.valence,
            "intensity": s.intensity,
            "tempo_local": s.tempo_local,
            "brightness": s.brightness,
        }
        for s in result.sections
    ]
    csv_path = out / f"{stem}_sections.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("[2/3] 生成可视化 …")
    fig_timeline(result).write_html(out / f"{stem}_timeline.html")
    fig_overview(result).write_html(out / f"{stem}_overview.html")
    fig_section_bars(result).write_html(out / f"{stem}_bars.html")

    print("[3/3] 完成")
    print(
        f"  时长 {format_time(result.duration)} | BPM {result.tempo:.1f} | "
        f"段落 {len(result.sections)} | 算法 {result.method}"
    )
    if result.style:
        print(
            f"  风格 {result.style.top_style_zh}/{result.style.top_style} "
            f"({result.style.confidence:.0%}) | {result.style.backend}"
        )
        print(f"  标签 {', '.join(result.style.tags)}")
    for s in result.sections:
        print(
            f"  #{s.index:02d} [{format_time(s.start)}–{format_time(s.end)}] "
            f"{s.label:16s} | {s.emotion_tag} | 强度={s.intensity:.2f}"
        )
    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")
    print(f"  HTML → {out}")


if __name__ == "__main__":
    main()
