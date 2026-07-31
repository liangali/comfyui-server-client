@echo off
setlocal EnableExtensions
REM Start ComfyUI listening on all interfaces so LAN clients can call the API.
REM Usage (on the B70 / ComfyUI host):
REM   server\start_comfyui_remote.bat
REM
REM Clients should use:  http://<this-machine-LAN-IP>:8188
REM Example: python run_qwen_image_remote.py --server 192.168.1.50:8188
REM
REM Layout assumption:
REM   <repo>\comfyui-server-client\server\start_comfyui_remote.bat   (this file)
REM   <repo>\llm-scaler\omni\comfyui_windows_setup\ComfyUI\main.py

set "REPO_ROOT=%~dp0..\.."
set "SETUP_DIR=%REPO_ROOT%\llm-scaler\omni\comfyui_windows_setup"
set "PY_DIR=%SETUP_DIR%\python_embeded"
set "PATH=%PY_DIR%;%PY_DIR%\Scripts;%PY_DIR%\Library\bin;%PATH%"
set PYTHONPATH=
set PYTHONHOME=

cd /d "%SETUP_DIR%\ComfyUI"
if not exist "main.py" (
  echo ERROR: ComfyUI main.py not found under "%SETUP_DIR%\ComfyUI"
  echo Expected portable install at:
  echo   %SETUP_DIR%
  exit /b 1
)

echo ============================================================
echo  ComfyUI REMOTE / LAN mode
echo  Listen: 0.0.0.0:8188  (all interfaces)
echo  Local GUI:  http://127.0.0.1:8188
echo  LAN GUI/API: http://^<this-PC-LAN-IP^>:8188
echo ============================================================
echo.
echo Showing IPv4 addresses on this machine (pick the LAN one):
ipconfig | findstr /i "IPv4"
echo.
echo If LAN clients cannot connect, allow inbound TCP 8188 in Windows Firewall:
echo   netsh advfirewall firewall add rule name="ComfyUI 8188" dir=in action=allow protocol=TCP localport=8188
echo.
echo Press Ctrl+C to stop the server.
echo ============================================================

"%PY_DIR%\python.exe" main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory %*
set "EC=%ERRORLEVEL%"
echo.
echo ComfyUI exited with code %EC%
exit /b %EC%
