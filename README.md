# comfyui-server-client

局域网远程调用 ComfyUI（Qwen-Image / Wan2.2 / SeedVR2）。

- **服务端（B70）**：启动 ComfyUI，监听 `0.0.0.0:8188`，加载模型并推理  
- **客户端（任意机器）**：用 Python 脚本提交任务；自动上传输入图/视频，下载输出图/视频  
- **执行方式**：串行（ComfyUI 队列一次只跑一个任务；脚本也是等上一个完成再继续）

```text
comfyui-server-client/
├── README.md
├── server/
│   └── start_comfyui_remote.bat      # 在 B70 上启动
└── client/
    ├── comfyui_remote_client.py      # 公共库（被下面脚本 import）
    ├── run_qwen_image_remote.py      # 文生图
    ├── run_wan22_remote.py           # 文/图生视频
    └── run_seedvr2_remote.py         # 视频超分
```

客户端只需 **Python 3.10+ 标准库**，不需要 GPU、不需要安装模型。

---

## 1. 服务端（B70）

### 1.1 前置

本机已装好 portable ComfyUI（PyTorch XPU），且 Qwen / Wan / SeedVR2 模型与节点可用。  
安装目录需在仓库内：

```text
<repo>/llm-scaler/omni/comfyui_windows_setup/ComfyUI/
```

### 1.2 启动

在 B70 上运行：

```bat
cd /d C:\data\code\llm_scaler_comfyui_code\comfyui-server-client
server\start_comfyui_remote.bat
```

启动参数：`--listen 0.0.0.0 --port 8188 --disable-smart-memory`。

终端会打印本机 IPv4，例如 `10.239.140.52`。**客户端要用这个局域网 IP**，不要用 `127.0.0.1`。

本机可打开：http://127.0.0.1:8188

### 1.3 防火墙（首次）

管理员 CMD / PowerShell：

```bat
netsh advfirewall firewall add rule name="ComfyUI 8188" dir=in action=allow protocol=TCP localport=8188
```

### 1.4 连通性检查（在客户端做）

把 IP 换成 B70 的局域网地址：

```bat
curl http://10.239.140.52:8188/system_stats
```

返回含 `devices` 的 JSON 即通。浏览器打开同一地址也应能看到 ComfyUI UI。

不通时检查：是否用了 `start_comfyui_remote.bat`、防火墙、是否同一网段、IP 是否选对（优先有线/Wi‑Fi 的 `192.168.x.x`）。

若本机设置了 `http_proxy`（常见于公司环境），普通 `curl` 会走代理并可能返回 **403 HTML（Policy ID）**，看起来像“连不上”。请用：

```bat
curl --noproxy "*" http://10.239.140.52:8188/system_stats
```

客户端 Python 脚本已强制直连、不走 `HTTP_PROXY`。也可把 B70 IP 加入 `no_proxy`：

```bat
set no_proxy=%no_proxy%,10.239.140.52
```

---

## 2. 客户端

### 2.1 准备

把整个 `client/` 目录拷到客户端任意位置（或直接用本仓库），进入该目录：

```bat
cd /d D:\path\to\comfyui-server-client\client
python --version
python -c "import comfyui_remote_client; print('ok')"
```

### 2.2 指定服务器

```bat
REM 方式 1：命令行（推荐）
python run_qwen_image_remote.py --server 10.239.140.52:8188

REM 方式 2：环境变量
set COMFYUI_SERVER=10.239.140.52:8188
python run_qwen_image_remote.py
```

优先级：`--server` > `COMFYUI_SERVER` > 默认 `127.0.0.1:8188`。

### 2.3 公共参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--server HOST:PORT` | B70 上 ComfyUI 地址 | 见上 |
| `--out-dir DIR` | 本地下载目录 | `./remote_outputs` |
| `--output PATH` | 主输出文件路径 | 自动命名到 `--out-dir` |
| `--ready-timeout SEC` | 等待服务就绪 | `60` |
| `--timeout SEC` | 单次推理超时 | Qwen `1800`；Wan/SeedVR2 `7200` |

每个脚本内部流程：探测服务 →（如有）上传输入 → 提交 `/prompt` → 轮询 `/history` → 用 `/view` 下载结果。

---

## 3. Qwen-Image（文生图）

```bat
cd /d D:\path\to\comfyui-server-client\client

REM 冒烟（默认 1024x1024）
python run_qwen_image_remote.py --server 10.239.140.52:8188

REM 自定义
python run_qwen_image_remote.py --server 10.239.140.52:8188 --prompt "cyberpunk rainy street guitarist, neon lights, cinematic" --width 1024 --height 1024 --seed 42 --output c:\data\qwen.png

REM 从文件读提示词 / 官方默认尺寸
python run_qwen_image_remote.py --server 10.239.140.52:8188 --prompt-file prompt.txt --output out.png
python run_qwen_image_remote.py --server 10.239.140.52:8188 --width 1328 --height 1328
```

| 参数 | 说明 |
|------|------|
| `--prompt` / `--prompt-file` | 正向提示词 |
| `--width` `--height` | 默认 `1024` |
| `--steps` | 默认 `10` |
| `--cfg` | 默认 `1.0` |
| `--seed` | 默认 `42` |

输出：PNG + `*_meta.json`（在 `--out-dir` 或 `--output`）。

---

## 4. Wan2.2（文/图生视频）

```bat
cd /d D:\path\to\comfyui-server-client\client

REM 14B I2V（最常用）：本地图上传 → 远端生成 → 下载 MP4
python run_wan22_remote.py --server 10.239.140.52:8188 --only i2v ^
  --image D:\inputs\hero.png ^
  --prompt "gentle camera push-in, soft wind, cinematic lighting" ^
  --width 640 --height 640 --duration 5 ^
  --output c:\data\hero_i2v.mp4

REM 14B T2V（纯文生视频，不需要 --image）
python run_wan22_remote.py --server 10.239.140.52:8188 --only t2v --output c:\data\t2v.mp4

REM 5B TI2V
python run_wan22_remote.py --server 10.239.140.52:8188 --only 5b --image D:\inputs\frame.png

REM 串行跑 5b + t2v + i2v（仍是一个接一个，不并发）
python run_wan22_remote.py --server 10.239.140.52:8188 --only all --image D:\inputs\frame.png
```

| 参数 | 说明 |
|------|------|
| `--only` | `5b` / `t2v` / `i2v` / `all`（默认 `i2v`） |
| `--image` | 本地输入图；`5b` / `i2v` / `all` 必填 |
| `--prompt` / `--prompt-file` | 正向提示词 |
| `--negative-prompt` / `--negative-prompt-file` | 负向提示词（有默认） |
| `--width` `--height` | I2V 分辨率，默认 `640`；建议为 8 的倍数 |
| `--duration` | I2V 目标秒数（@16fps） |
| `--seed` | 默认随机 |

`--only all` 且指定 `--output foo.mp4` 时，会写成 `foo_wan22_5b_ti2v.mp4` 等，避免覆盖。

输出：MP4 + `wan22_remote_summary_*.json`。

---

## 5. SeedVR2（视频超分）

```bat
cd /d D:\path\to\comfyui-server-client\client

python run_seedvr2_remote.py --server 10.239.140.52:8188 ^
  --video c:\data\hero_i2v.mp4 ^
  --resolution 1080 ^
  --output c:\data\hero_1080p.mp4
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--video` | （必填） | 本地输入视频 |
| `--resolution` | `1080` | 目标**高度**（如 832×480 → 约 1872×1080） |
| `--batch-size` | `33` | |
| `--blocks-to-swap` | `32` | |
| `--seed` | `42` | |
| `--skip-node-check` | off | 跳过远端 SeedVR2 节点检查 |

输出：MP4 + `seedvr2_remote_meta_*.json`。

---

## 6. 串联示例（文生图 → 图生视频 → 超分）

B70 上保持 `start_comfyui_remote.bat` 运行，客户端串行执行：

```bat
cd /d D:\path\to\comfyui-server-client\client
set SERVER=10.239.140.52:8188
set OUT=D:\pipeline_out
mkdir %OUT%

python run_qwen_image_remote.py --server %SERVER% --out-dir %OUT% ^
  --prompt "cyberpunk rainy street guitarist, neon reflections, cinematic still" ^
  --width 1024 --height 1024 --output %OUT%\frame.png

python run_wan22_remote.py --server %SERVER% --out-dir %OUT% --only i2v ^
  --image %OUT%\frame.png ^
  --prompt "the musician gently plays guitar, camera slowly pushes in, rain falls, cinematic" ^
  --width 832 --height 480 --duration 10 ^
  --output %OUT%\clip_480p.mp4

python run_seedvr2_remote.py --server %SERVER% --out-dir %OUT% ^
  --video %OUT%\clip_480p.mp4 --resolution 1080 ^
  --output %OUT%\clip_1080p.mp4
```

---

## 7. 注意事项

1. **串行**：单卡只适合一次跑一个重任务。多客户端同时提交会进入 ComfyUI 队列排队，不会并行占满算力，但会互相等待。
2. **无鉴权**：`0.0.0.0:8188` 对局域网开放，仅用于可信内网；不要映射到公网。
3. **模型与节点在 B70**：客户端只调 API；缺模型/节点时看 B70 终端日志。
4. **排错优先看 B70 终端**：加载、采样、`Prompt executed in Xs` 都在那边。

| 现象 | 处理 |
|------|------|
| `Cannot reach ComfyUI` / `not reachable` | 确认 remote 启动脚本、防火墙、`--server IP:8188`；公司代理下用 `curl --noproxy "*"` 测通 |
| `curl` 返回 HTML / Policy ID / 403 | 请求被 `http_proxy` 拦截；加 `--noproxy "*"`，Python 脚本已默认直连 |
| `node_errors` / 缺节点 | 在 B70 上确认模型与自定义节点已安装 |
| SeedVR2 `nodes not loaded` | B70 上安装并启用 SeedVR2 节点 |
| 超时 | 加大 `--timeout`；检查 B70 是否卡死或排队过长 |
| OOM | 降低分辨率或 `--duration` |
