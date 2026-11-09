@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 音乐段落情绪分析
echo.
echo  ========================================
echo   音乐段落情绪分析
echo  ========================================
echo.

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

if not defined PY (
  echo [错误] 未找到 Python。请安装 Python 3.10+ 并勾选 Add to PATH。
  pause
  exit /b 1
)

%PY% launch.py
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo.
  echo [错误] 退出码 %ERR%
  pause
)
exit /b %ERR%
