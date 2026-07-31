# comfyui-server-client

> 适用：Windows 11 + Intel Arc Pro B70（服务端） + 同局域网任意客户端机器  
> 目标：在 B70 上启动可局域网访问的 ComfyUI，从另一台机器远程调用 **Qwen-Image / Wan2.2 / SeedVR2**，并自动上传输入、下载输出

---

## 1. 结论摘要

| 问题 | 答案 |
|------|------|
| ComfyUI 是否支持局域网调用？ | **支持**。原生 HTTP + WebSocket API，把 `--listen` 改为 `0.0.0.0` 即可 |
| 模型权重放哪？ | **只在 B70**。客户端不需要 GPU / 不需要下载模型 |
| 客户端需要什么？ | Python 3.10+（标准库即可），拷贝本仓库里的 remote 脚本 |
| 输入输出怎么传？ | 脚本自动：本地文件 → `POST /upload/image` → 推理 → `GET /view` → 写回本地 |

**架构：**

```text
[客户端机器]                         [B70 机器]
  run_*_remote.py  --server IP:8188
       |  upload image/video
       |  POST /prompt
       |  poll /history
       |  download /view
       v
                              ComfyUI --listen 0.0.0.0:8188
                              Qwen / Wan / SeedVR2 在 XPU 上推理
                              模型与 output 均在 B70 本地盘
```

---

## 2. 新增文件一览

| 文件 | 角色 | 运行位置 |
|------|------|----------|
| `start_comfyui_remote.bat` | 以 `0.0.0.0:8188` 启动 ComfyUI | **B70** |
| `comfyui_remote_client.py` | 共享客户端库（上传/排队/下载） | 客户端（被 import） |
| `run_qwen_image_remote.py` | 远程文生图 | 客户端 |
| `run_wan22_remote.py` | 远程文/图生视频 | 客户端 |
| `run_seedvr2_remote.py` | 远程视频超分 | 客户端 |
| `COMFYUI_REMOTE_LAN_GUIDE.md` | 本文档 | — |

> 原有本机脚本（`start_comfyui_qwen.bat`、`run_qwen_image.py` 等）**保持不变**，仍绑定 `127.0.0.1`，适合本机调试。

---

## 3. B70 服务端准备（一次性 + 每次开机）

### 3.1 前置条件

确认本机已按 `QWEN_IMAGE_B70_SETUP_AND_RUN.md` 完成：

- portable ComfyUI + PyTorch XPU
- Qwen-Image / Wan2.2 / SeedVR2 模型与自定义节点
- 本机冒烟测试曾经成功（`127.0.0.1:8188`）

### 3.2 启动局域网服务

在 **B70 机器**上双击或命令行运行：

```bat
C:\data\code\llm_scaler_comfyui_code\start_comfyui_remote.bat
```

与本机版的关键差别：

| | 本机 `start_comfyui_qwen.bat` | 远程 `start_comfyui_remote.bat` |
|--|--|--|
| listen | `127.0.0.1`（仅本机） | `0.0.0.0`（所有网卡） |
| 局域网可访问 | 否 | **是** |

启动后终端会打印本机 IPv4 列表，例如：

```text
IPv4 Address. . . . . . . . . . . : 192.168.1.50
```

记下这个 **局域网 IP**（不要用 `127.0.0.1`，客户端用不了）。

成功标志：

```text
Starting server
To see the GUI go to: http://0.0.0.0:8188
```

本机浏览器仍可打开：http://127.0.0.1:8188

### 3.3 放行 Windows 防火墙（首次必须）

以**管理员** PowerShell / CMD 执行：

```bat
netsh advfirewall firewall add rule name="ComfyUI 8188" dir=in action=allow protocol=TCP localport=8188
```

或在「Windows 安全中心 → 防火墙 → 高级设置 → 入站规则」中新建规则：TCP 端口 `8188`，允许连接，专用网络。

### 3.4 在客户端先做连通性检查

在**另一台机器**的浏览器或终端：

```bat
curl http://192.168.1.50:8188/system_stats
```

把 `192.168.1.50` 换成你的 B70 IP。返回 JSON（含 `devices` / VRAM）即表示通。

浏览器也可直接打开：

```text
http://192.168.1.50:8188
```

应看到 ComfyUI Web UI。

**若不通，按顺序排查：**

1. B70 上是否用的是 `start_comfyui_remote.bat`（不是旧的 `127.0.0.1` 脚本）
2. 防火墙是否放行 8188
3. 两台机器是否同一网段 / 未开访客隔离（部分公司 Wi‑Fi 禁止终端互访）
4. `ipconfig` 是否看错了虚拟网卡 IP（优先用以太网/Wi‑Fi 的 192.168.x.x）

---

## 4. 客户端准备

### 4.1 需要拷贝的文件

在客户端任意目录（例如 `D:\comfy_remote_client\`）放置：

```text
comfyui_remote_client.py
run_qwen_image_remote.py
run_wan22_remote.py
run_seedvr2_remote.py
```

也可直接把整个 `llm_scaler_comfyui_code` 仓库拷过去/挂载过去；客户端**不需要** `comfyui_windows_setup`、模型和 GPU。

### 4.2 Python 要求

- Python **3.10+**（推荐 3.11/3.12）
- **仅标准库**：`urllib` / `json` / `argparse` / `pathlib` 等
- **不需要** PyTorch、不需要 XPU、不需要安装 ComfyUI

验证：

```bat
python --version
python -c "import comfyui_remote_client; print('ok')"
```

（需在上述脚本所在目录执行。）

### 4.3 指定服务器地址的两种方式

**方式 1：命令行（推荐）**

```bat
python run_qwen_image_remote.py --server 192.168.1.50:8188
```

**方式 2：环境变量**

```bat
set COMFYUI_SERVER=192.168.1.50:8188
python run_qwen_image_remote.py
```

优先级：`--server` > `COMFYUI_SERVER` > 默认 `127.0.0.1:8188`。

---

## 5. 公共参数说明

三个 `run_*_remote.py` 都支持：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--server HOST:PORT` | B70 上 ComfyUI 地址 | `127.0.0.1:8188` 或环境变量 |
| `--out-dir DIR` | 本地下载输出目录 | `./remote_outputs` |
| `--ready-timeout SEC` | 等待服务就绪超时 | `60` |
| `--timeout SEC` | 单次推理等待超时 | Qwen 1800 / Wan&SeedVR2 7200 |
| `--output PATH` | 指定主输出文件本地路径 | 自动命名到 `--out-dir` |

完整流程（每个脚本内部自动完成）：

1. `GET /system_stats` — 确认可达并打印远端 GPU
2. （若有输入）`POST /upload/image` — 上传图片/视频到 B70 的 `input/`
3. `POST /prompt` — 提交与本机验证过的等价工作流
4. 轮询 `GET /history/{prompt_id}` — 直到有 outputs
5. `GET /view?...` — 把图片/视频下载到客户端 `--out-dir`
6. 写一份 JSON meta / summary

---

## 6. Qwen-Image 远程调用

### 6.1 最简示例

```bat
cd /d D:\comfy_remote_client
python run_qwen_image_remote.py --server 192.168.1.50:8188
```

默认：1024×1024、10 steps、distill 模型、内置英文测试提示词。  
输出：`remote_outputs/qwen_image_1024x1024_YYYYMMDD_HHMMSS.png`

### 6.2 自定义提示词与分辨率

```bat
python run_qwen_image_remote.py ^
  --server 192.168.1.50:8188 ^
  --prompt "cyberpunk rainy street guitarist, neon lights, cinematic" ^
  --width 1024 --height 1024 ^
  --seed 20260731 ^
  --output D:\outs\my_qwen.png
```

或从文件读提示词：

```bat
python run_qwen_image_remote.py --server 192.168.1.50:8188 --prompt-file prompt.txt --output out.png
```

### 6.3 官方默认 1328²

```bat
python run_qwen_image_remote.py --server 192.168.1.50:8188 --width 1328 --height 1328
```

### 6.4 预期耗时（参考 B70）

| 分辨率 | 冷启动（含加载） | 热模型 |
|--------|------------------|--------|
| 1024² | ~55–70 s | ~35–45 s |
| 1328² | — | ~35–45 s |

Peak VRAM ≈ **19.3 GB**（模型在 B70 上）。

---

## 7. Wan2.2 远程调用

### 7.1 14B I2V（最常用：本地图 → 远端生成视频 → 下载 MP4）

```bat
python run_wan22_remote.py ^
  --server 192.168.1.50:8188 ^
  --only i2v ^
  --image D:\inputs\hero.png ^
  --prompt "gentle camera push-in, soft wind, cinematic lighting" ^
  --width 640 --height 640 --duration 5 ^
  --output D:\outs\hero_i2v.mp4
```

脚本会：

1. 把 `hero.png` 上传到 B70
2. 跑 Wan2.2 14B I2V + lightx2v 4-step LoRA
3. 把 MP4 下载到 `D:\outs\hero_i2v.mp4`

### 7.2 14B T2V（纯文生视频，无需 `--image`）

```bat
python run_wan22_remote.py --server 192.168.1.50:8188 --only t2v --output D:\outs\t2v.mp4
```

### 7.3 5B TI2V（图+文，832×480）

```bat
python run_wan22_remote.py --server 192.168.1.50:8188 --only 5b --image D:\inputs\frame.png
```

### 7.4 一次跑三种（`all`）

```bat
python run_wan22_remote.py --server 192.168.1.50:8188 --only all --image D:\inputs\frame.png
```

多任务时若指定 `--output foo.mp4`，会自动写成 `foo_wan22_5b_ti2v.mp4` 等，避免互相覆盖。

### 7.5 参数表

| 参数 | 说明 |
|------|------|
| `--only` | `5b` / `t2v` / `i2v` / `all`（默认 `i2v`） |
| `--image` | 本地输入图（`5b`/`i2v`/`all` 必填） |
| `--prompt` / `--prompt-file` | 正向提示词 |
| `--negative-prompt` / `--negative-prompt-file` | 负向提示词（默认内置中文 NEG） |
| `--width` `--height` | I2V 分辨率（默认 640×640；建议 8 的倍数） |
| `--duration` | I2V 目标秒数（@16fps，帧数会对齐 Wan 约束） |
| `--seed` | 噪声种子（默认随机） |

### 7.6 预期耗时（参考 B70）

| 模式 | 典型墙钟 |
|------|----------|
| 5B TI2V ~3s 832×480 | ~70 s |
| 14B T2V 5s 640² 4-step | ~170 s |
| 14B I2V 5s 640² 4-step | ~180 s |
| 14B I2V 10s 832×480 | ~380 s |

---

## 8. SeedVR2 远程调用

### 8.1 上传本地视频并超分到 1080p

```bat
python run_seedvr2_remote.py ^
  --server 192.168.1.50:8188 ^
  --video D:\outs\hero_i2v.mp4 ^
  --resolution 1080 ^
  --output D:\outs\hero_1080p.mp4
```

流程：本地 MP4 → 上传 B70 `input/` → SeedVR2（XPU）→ 下载超分 MP4。

### 8.2 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--video` | （必填） | 本地输入视频 |
| `--resolution` | 1080 | SeedVR2 目标**高度** |
| `--batch-size` | 33 | 与本机验证配置一致 |
| `--blocks-to-swap` | 32 | XPU block swap |
| `--seed` | 42 | |
| `--skip-node-check` | off | 跳过远端 SeedVR2 节点探测 |

### 8.3 预期耗时（参考：10s 480p → 1080p）

| 指标 | 值 |
|------|-----|
| 墙钟 | ~416 s |
| Peak VRAM | ~9.5–9.7 GB |

> `resolution=1080` 按高度定义；例如 832×480 会得到约 **1872×1080**。

---

## 9. 端到端串联示例（客户端一气呵成）

在客户端依次执行（B70 上 `start_comfyui_remote.bat` 保持运行）：

```bat
set SERVER=192.168.1.50:8188
set OUT=D:\pipeline_out
mkdir %OUT%

REM 1) 文生图
python run_qwen_image_remote.py --server %SERVER% --out-dir %OUT% ^
  --prompt "cyberpunk rainy street guitarist, neon reflections, cinematic still" ^
  --width 1024 --height 1024 --output %OUT%\frame.png

REM 2) 图生视频 10s 480p
python run_wan22_remote.py --server %SERVER% --out-dir %OUT% --only i2v ^
  --image %OUT%\frame.png ^
  --prompt "the musician gently plays guitar, camera slowly pushes in, rain falls, cinematic" ^
  --width 832 --height 480 --duration 10 ^
  --output %OUT%\clip_480p.mp4

REM 3) 超分 1080p
python run_seedvr2_remote.py --server %SERVER% --out-dir %OUT% ^
  --video %OUT%\clip_480p.mp4 --resolution 1080 ^
  --output %OUT%\clip_1080p.mp4
```

总墙钟参考（B70 实测本机串联量级）：约 **14–15 分钟**（含模型加载；热启动会略快）。

---

## 10. 输出文件约定

默认下载目录：脚本旁的 `remote_outputs/`（可用 `--out-dir` 改）。

| 脚本 | 典型输出 |
|------|----------|
| Qwen | `qwen_image_{W}x{H}_{stamp}.png` + `*_meta.json` |
| Wan | `{model}_{W}x{H}_{dur}s_{stamp}.mp4` + `wan22_remote_summary_*.json` |
| SeedVR2 | `seedvr2_{res}p_{stamp}.mp4` + `seedvr2_remote_meta_*.json` |

History 里的原始文件名也会先落到 `--out-dir`，再复制一份友好文件名；以 `--output` 指定的路径为准作为“主产物”。

---

## 11. 安全与使用注意

1. **无鉴权**：`0.0.0.0:8188` 对局域网内任何人开放 API（可提交任意工作流、读写 input/output）。仅在**可信内网**使用。
2. **不要**把 8188 端口映射到公网。
3. 用完可改回本机模式：停止 remote bat，改用 `start_comfyui_qwen.bat`（只听 `127.0.0.1`）。
4. 单卡 B70 **同一时间只适合跑一个重任务**；多客户端同时提交会排队，显存仍可能互相挤兑。
5. 大视频上传/下载受局域网带宽影响；千兆局域网通常足够（几十 MB 级 MP4）。
6. SeedVR2 / Wan 节点必须已在 **B70** 上安装好；客户端只发 API，不会帮你装节点。

---

## 12. 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `Cannot reach ComfyUI` | 服务未启 / 仍监听 127.0.0.1 / 防火墙 | 用 `start_comfyui_remote.bat`；放行 8188；`curl /system_stats` |
| 浏览器能开 UI，脚本失败 | `--server` 写错端口或漏写 | 确认 `IP:8188` |
| `prompt submit failed` / `node_errors` | 远端缺模型或节点名不匹配 | 在 B70 上先跑通本机脚本 |
| SeedVR2：`nodes not loaded` | 自定义节点未装或未打 XPU 补丁 | 按 `QWEN_IMAGE_B70_SETUP_AND_RUN.md` §17 |
| 上传成功但 LoadImage 找不到文件 | 极少见的重名改名 | 脚本已用唯一 `remote_*` 文件名；看上传返回的 `name` |
| 超时 | 任务太长或服务卡死 | 加大 `--timeout`；看 B70 终端日志 |
| OOM | 分辨率/时长过大 | 降分辨率；Wan 用更短 `--duration` |

B70 终端日志仍是排错第一现场：模型加载、采样进度、`Prompt executed in Xs` 都会打在那里。

---

## 13. 与本机脚本的对照

| 能力 | 本机脚本 | 远程脚本 |
|------|----------|----------|
| Qwen | `run_qwen_image.py` | `run_qwen_image_remote.py` |
| Wan | `run_wan22.py` | `run_wan22_remote.py` |
| SeedVR2 | `run_seedvr2_standalone.py` | `run_seedvr2_remote.py` |
| 启动 | `start_comfyui_qwen.bat` | `start_comfyui_remote.bat` |
| 输入 | 直接 `copy` 到 `ComfyUI/input` | HTTP multipart 上传 |
| 输出 | 从 `ComfyUI/output` 本地 copy | HTTP `/view` 下载到客户端 |

工作流节点图与本机验证版一致（同一套模型文件名与采样参数），保证行为可预期。

---

## 14. 快速检查清单

**B70：**

- [ ] 模型与节点已装好，本机曾跑通
- [ ] 运行 `start_comfyui_remote.bat`
- [ ] 记下局域网 IPv4
- [ ] 防火墙放行 TCP 8188
- [ ] 本机打开 http://127.0.0.1:8188 正常

**客户端：**

- [ ] 已拷贝 4 个 `.py` 文件（含 `comfyui_remote_client.py`）
- [ ] `curl http://<B70-IP>:8188/system_stats` 成功
- [ ] `python run_qwen_image_remote.py --server <B70-IP>:8188` 出图并落到 `remote_outputs/`
- [ ] （可选）Wan I2V / SeedVR2 上传下载闭环验证

---

## 15. 相关文档

- 本机安装与性能基线：`QWEN_IMAGE_B70_SETUP_AND_RUN.md`
- llm-scaler ComfyUI 中文指南：`llm-scaler/omni/docs/ComfyUI_Guide_CN.md`
- ComfyUI 官方 API 示例：`ComfyUI/script_examples/websockets_api_example.py`
