# comfyui-server-client

局域网远程调用 ComfyUI（Qwen-Image / Wan2.2 / SeedVR2 / LTX-2.3）。

- **服务端（B70）**：启动 ComfyUI，监听 `0.0.0.0:8188`，加载模型并推理  
- **客户端（任意机器）**：用 Python 脚本提交任务；自动上传输入图/视频，下载输出图/视频  
- **执行方式**：串行（ComfyUI 队列一次只跑一个任务；脚本也是等上一个完成再继续）

```text
comfyui-server-client/
├── README.md
├── server/
│   ├── start_comfyui_remote.bat      # 在 B70 上启动（Windows）
│   └── start_comfyui_remote.sh       # 在 Linux 主机上启动
└── client/
    ├── comfyui_remote_client.py      # 公共库（被下面脚本 import）
    ├── run_qwen_image_remote.py      # 文生图
    ├── run_wan22_remote.py           # 文/图生视频
    ├── run_seedvr2_remote.py         # 视频超分
    └── run_ltx23_i2v_remote.py       # LTX-2.3 图生视频
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

终端会打印本机 IPv4，例如 `192.168.1.50`。**客户端要用这个局域网 IP**，不要用 `127.0.0.1`。

本机可打开：http://127.0.0.1:8188

Linux 主机（conda + torch-xpu）用 `.sh` 版本，参数与 `.bat` 一致：

```bash
server/start_comfyui_remote.sh
# 换端口 / 换 ComfyUI 目录 / 换 ComfyUI 启动参数
COMFY_PORT=8189 COMFYUI_DIR=/path/to/ComfyUI COMFY_FLAGS="--reserve-vram 2" \
  server/start_comfyui_remote.sh
```

默认在 `<repo>/../ComfyUI` 和 llm-scaler 的 portable 目录里找 `main.py`，找到名为
`comfyui` 的 conda 环境就自动激活（`CONDA_ENV` 可改），启动时打印本机局域网地址。

### 1.3 防火墙（首次）

管理员 CMD / PowerShell：

```bat
netsh advfirewall firewall add rule name="ComfyUI 8188" dir=in action=allow protocol=TCP localport=8188
```

### 1.4 连通性检查（在客户端做）

把 IP 换成 B70 的局域网地址：

```bat
curl http://192.168.1.50:8188/system_stats
```

返回含 `devices` 的 JSON 即通。浏览器打开同一地址也应能看到 ComfyUI UI。

不通时检查：是否用了 `start_comfyui_remote.bat`、防火墙、是否同一网段、IP 是否选对（优先有线/Wi‑Fi 的 `192.168.x.x`）。

若本机设置了 `http_proxy`（常见于公司环境），普通 `curl` 会走代理并可能返回 **403 HTML（Policy ID）**，看起来像“连不上”。请用：

```bat
curl --noproxy "*" http://192.168.1.50:8188/system_stats
```

客户端 Python 脚本已强制直连、不走 `HTTP_PROXY`。也可把 B70 IP 加入 `no_proxy`：

```bat
set no_proxy=%no_proxy%,192.168.1.50
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
python run_qwen_image_remote.py --server 192.168.1.50:8188

REM 方式 2：环境变量
set COMFYUI_SERVER=192.168.1.50:8188
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
python run_qwen_image_remote.py --server 192.168.1.50:8188

REM 自定义
python run_qwen_image_remote.py --server 192.168.1.50:8188 --prompt "cyberpunk rainy street guitarist, neon lights, cinematic" --width 1024 --height 1024 --seed 42 --output c:\data\qwen.png

REM 从文件读提示词 / 官方默认尺寸
python run_qwen_image_remote.py --server 192.168.1.50:8188 --prompt-file prompt.txt --output out.png
python run_qwen_image_remote.py --server 192.168.1.50:8188 --width 1328 --height 1328
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
python run_wan22_remote.py --server 192.168.1.50:8188 --only i2v --image c:\data\qwen.png --prompt "gentle camera push-in, soft wind, cinematic lighting" --width 640 --height 640 --duration 5 --output c:\data\qwen_i2v.mp4

REM 14B T2V（纯文生视频，不需要 --image）
python run_wan22_remote.py --server 192.168.1.50:8188 --only t2v --output c:\data\t2v.mp4

REM 5B TI2V
python run_wan22_remote.py --server 192.168.1.50:8188 --only 5b --image c:\data\frame.png

REM 串行跑 5b + t2v + i2v（仍是一个接一个，不并发）
python run_wan22_remote.py --server 192.168.1.50:8188 --only all --image c:\data\frame.png
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

python run_seedvr2_remote.py --server 192.168.1.50:8188 --video c:\data\qwen_i2v.mp4 --resolution 1080 --output c:\data\qwen_1080p.mp4
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

## 6. LTX-2.3（图生视频，带音轨）

```bash
cd /path/to/comfyui-server-client/client

# 冒烟：低分辨率 + 跳过精修，最快出片
python run_ltx23_i2v_remote.py --server 192.168.1.50:8188 \
  --image /data/frame.png --width 512 --height 288 --duration 1 --no-refine

# 常用：1280x704 / 5 秒 / 25fps
python run_ltx23_i2v_remote.py --server 192.168.1.50:8188 \
  --image /data/frame.png \
  --prompt "the camera slowly pushes in, subtle natural motion, cinematic lighting" \
  --width 1280 --height 720 --duration 5 --fps 25 \
  --output /data/frame_i2v.mp4

# 只导出 API 格式 prompt，不连服务器（排错用）
python run_ltx23_i2v_remote.py --image x.png --dump ltx_i2v_api.json
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image` | （必填） | 本地首帧图，自动上传到服务端 |
| `--prompt` / `--prompt-file` | 有默认 | 正向提示词 |
| `--negative-prompt` / `--negative-prompt-file` | 有默认 | 负向提示词 |
| `--width` `--height` | `1280` `720` | 最终分辨率，自动对齐到 64 的倍数（见下） |
| `--duration` | `5` | 目标秒数；帧数 = `duration * fps + 1` |
| `--fps` | `25` | |
| `--seed` | 随机 | 基础采样噪声种子（精修阶段固定 42，与官方模板一致） |
| `--no-refine` | off | 只跑基础采样：输出是一半分辨率，快很多 |
| `--img-compression` | `18` | `LTXVPreprocess` 对首帧的压缩强度 |
| `--distill-strength` | `0.5` | 蒸馏 LoRA 强度 |
| `--ckpt` `--text-encoder` `--distill-lora` `--upscaler` | 见下 | 覆盖模型文件名 |
| `--skip-model-check` | off | 跳过提交前的节点/模型名检查 |
| `--dump FILE` | — | 只写出 API 格式 prompt 后退出 |

输出：MP4（含 LTX 生成的音轨）+ `ltx23_i2v_meta_*.json`。

服务端需要的模型：`checkpoints/ltx-2.3-22b-dev-fp8.safetensors`、
`text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`、
`loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors`、
以及 `LatentUpscaleModelLoader` 提供的 `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`。
提交前脚本会查 `/object_info`：节点类缺失直接报错；模型名不在服务端列表里时退到最接近的
同名文件（例如 upscaler `x2-1.1` → `x2-1.0`）并打印提示。

**分辨率会被对齐**：LTX 的基础 latent 向下取整到 32 px 的倍数，而基础采样跑一半分辨率，
所以不是 64 倍数的尺寸会被服务端悄悄缩掉（`1280x720` 实际出 `1280x704`）。脚本在提交前
就四舍五入到 64 的倍数并打印提示，保证居中裁剪、日志、文件名、meta 和真正出来的文件一致。
`--no-refine` 时输出是这个尺寸的一半（`1280x704` → `640x352`）。

**关于工作流**：图来自 ComfyUI 自带模板 `video_ltx2_3_i2v.json`。该模板把整条链路包在一个
**UI subgraph** 里，而后端只展开节点级动态 subgraph、不展开 UI subgraph 定义，所以
`/prompt` 不能直接吃这个文件。脚本里的 `build_prompt()` 把它摊平成 API 格式，节点编号与模板
保持一致（269 / 75 / 276..328）方便对照。两段式采样：基础段把首帧经
`ResizeImageMaskNode`（居中裁剪）→ `ResizeImagesByLongerEdge(1536)` → `LTXVPreprocess`
写进 `(width/2, height/2)` 的空 latent（`LTXVImgToVideoInplace`，strength 0.7），8 步采样；
精修段用 `LTXVLatentUpsampler` x2 放大 latent 后再 3 步采样。

相对模板有三处有意改动：

1. 文本编码器默认 `gemma_3_12B_it_fp8_scaled`，不是模板的 `gemma_3_12B_it_fp4_mixed`
   ——NVFP4 走 comfy_kitchen，其原生后端只有 CUDA，在 Arc 上没有加速；可用 `--text-encoder` 改回。
2. 去掉 Gemma 提示词增强分支（`LoraLoader` + `TextGenerateLTX2Prompt` + `ComfySwitchNode`），
   `--prompt` 原样使用。模板里这个开关默认也是关的，开了每次多跑一遍 LLM。
3. 模板里的 `PrimitiveInt` / `ComfyMathExpression` 全部常量折叠掉了。

---

## 7. 串联示例（文生图 → 图生视频 → 超分）

B70 上保持 `start_comfyui_remote.bat` 运行，客户端串行执行：

```bat
cd /d D:\path\to\comfyui-server-client\client
set SERVER=192.168.1.50:8188
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

## 8. 注意事项

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
| LTX `missing node classes` | 服务端 ComfyUI 太旧或缺 LTX-2.3 节点，升级后重启 |
| 超时 | 加大 `--timeout`；检查 B70 是否卡死或排队过长 |
| OOM | 降低分辨率或 `--duration` |
