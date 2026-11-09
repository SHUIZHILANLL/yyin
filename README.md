# 音乐段落 · 情绪强度分析（yyin）

上传一首歌，自动划分曲式段落，估计情绪（效价 / 唤醒 / 强度）与风格，并在播放时用时间线、全曲曲线和拍点色条同步展示。

## 能做什么

1. **结构分段**  
   谱聚类 `scluster`（推荐）、Foote 新奇度、混合 `hybrid`，输出 Intro / Verse / Chorus 等段落时间轴。

2. **情绪指数**  
   基于声学特征映射 Russell 效价–唤醒空间，并合成**情绪强度**曲线与段级分数。  
   后处理参考 MediaEval / DEAM：长窗平滑 + 软归一（z→tanh），段间相对对比，避免指数暴涨暴跌。

3. **风格标签**  
   GTZAN 风格启发式（可选接 Hubert）；分阶段加载，风格算完**不打断**原曲播放。

4. **同步播放器**  
   原曲 + 曲式色条 + 波形 / 能量 / 唤醒·效价 / 强度曲线，红线跟播。  
   顶部 **BEAT** 色条：按稳定视觉拍网格闪光，**绿=强度低、红=强度高**，亮度一致。

5. **一键启动**  
   Windows 双击 `启动.bat`，或 `python launch.py`（默认 <http://127.0.0.1:8501>）。

## 快速开始

```bash
pip install -r requirements.txt
python launch.py
# 或
streamlit run app.py
```

左侧上传 mp3/wav/flac 等 → **① 开始核心分析** → 可播放；风格可自动或点 **② 分析风格**。

## 主要文件

| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit 界面（分阶段 + fragment） |
| `analyzer.py` | 核心分析流水线、稳定拍点提取 |
| `structure.py` | Foote / 谱聚类分段 |
| `emotion.py` | 情绪曲线稳定化与段级软缩放 |
| `style.py` | 风格估计 |
| `sync_player.py` | 同步时间线 + BEAT 闪色 |
| `launch.py` / `启动.bat` | 启动器（含端口占用与依赖修复） |

## 依赖说明

- 需 `uvicorn>=0.52` 且 `websockets<17`，否则 Streamlit 可能一直骨架屏。  
- 可选：安装 `torch` + `transformers` 可启用 Hubert GTZAN 风格模型。

## 分析采样率

界面中的「分析采样率」只影响特征提取用的重采样（默认 22050 Hz），与试听音质无关。

## License

按仓库所有者约定使用。
