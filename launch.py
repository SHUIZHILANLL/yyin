"""音乐段落情绪分析 — 一键启动器。"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
REQ = ROOT / "requirements.txt"
DEFAULT_PORT = int(os.environ.get("YYIN_PORT", "8501"))


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def _pick_port(start: int = DEFAULT_PORT, span: int = 20) -> int:
    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"端口 {start}-{start + span - 1} 均不可用")


def _http_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=1.5) as r:
            return 200 <= getattr(r, "status", 200) < 500
    except Exception:
        return False


def _fix_stream_stack() -> None:
    """websockets 17 + uvicorn<0.52 会导致页面一直骨架屏。"""
    try:
        import uvicorn
        import websockets

        uv = tuple(int(x) for x in uvicorn.__version__.split(".")[:2])
        ws_major = int(websockets.__version__.split(".", 1)[0])
        if uv >= (0, 52) and ws_major < 17:
            return
    except Exception:
        pass
    print("[yyin] 修复 Streamlit 通信依赖（uvicorn / websockets）…")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "uvicorn>=0.52.0",
            "websockets>=16.1.1,<17",
        ],
        cwd=str(ROOT),
    )


def _ensure_deps() -> None:
    try:
        import streamlit  # noqa: F401
        import librosa  # noqa: F401
        import plotly  # noqa: F401
    except ImportError:
        print("[yyin] 缺少依赖，正在安装 requirements.txt …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQ)],
            cwd=str(ROOT),
        )
    _fix_stream_stack()


def _wait_ready(url: str, seconds: float = 45.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _http_ok(url):
            return True
        time.sleep(0.4)
    return False


def main() -> int:
    os.chdir(ROOT)
    if not APP.is_file():
        print(f"[yyin] 找不到 {APP}")
        return 1

    home = f"http://127.0.0.1:{DEFAULT_PORT}"
    # 已有实例：直接打开浏览器
    if _port_open(DEFAULT_PORT) and _http_ok(home):
        print(f"[yyin] 服务已在运行 → {home}")
        webbrowser.open(home)
        print("[yyin] 已打开浏览器。如要重启，先关掉旧窗口再点启动。")
        return 0

    _ensure_deps()
    port = DEFAULT_PORT if not _port_open(DEFAULT_PORT) else _pick_port(DEFAULT_PORT + 1)
    url = f"http://127.0.0.1:{port}"
    if port != DEFAULT_PORT:
        print(f"[yyin] {DEFAULT_PORT} 被占用，改用 {port}")

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"[yyin] 启动中 → {url}")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)

    if _wait_ready(url):
        print(f"[yyin] 就绪，打开浏览器：{url}")
        webbrowser.open(url)
    else:
        print(f"[yyin] 等待超时，请手动打开：{url}")

    print("[yyin] 关闭本窗口即停止服务。Ctrl+C 也可结束。")
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(f"[yyin] 启动失败：{exc}")
        code = 1
    if code != 0:
        try:
            input("按回车键退出…")
        except EOFError:
            pass
    raise SystemExit(code)
