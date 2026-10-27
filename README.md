# 桌面伴侣 · 语音服务

> **本项目从设计、编码、调试到文档，全程由 AI（[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)）编写完成。**

**为桌面宠物适配的常驻语音后端**。
基于 [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) + 自定义音色 LoRA，对外暴露 OpenAI 兼容的语音接口。

> 🔗 **配套桌宠项目：[桌面伴侣 · Desktop Companion](https://github.com/crsm5192/desktop-companion)**
> —— 它是本服务的**使用方**：桌宠负责 Live2D 渲染、对话、音乐、番茄钟等，说话时调用本服务拿音频；
> 本项目只负责「把文字变成那个角色的声音」。两者通过 HTTP 解耦，可分别部署、甚至跑在不同机器上。

## 它解决什么问题

桌宠每次说话都要合成语音。如果每次调用都重新加载模型，会出现：

- 每次合成前要等几十秒加载权重 → 对话卡顿、毫无陪伴感
- 反复加载/卸载模型 → 显存碎片、OOM

本项目把模型**常驻在内存里只加载一次**，之后每次请求只是推理：

- 空闲时模型放 CPU（不占显存，把显卡还给其他程序）
- 合成时自动搬上 GPU，用完 `gpu-idle-seconds` 秒后自动退回 CPU
- 桌宠端只需一次 HTTP POST 就能拿到 WAV/MP3

## 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 健康检查，返回当前设备（cpu/cuda）与语言 |
| `POST` | `/tts` | 简单合成：`{"text","lang"?,"speed"?}` → `audio/wav` |
| `POST` | `/v1/audio/speech` | **OpenAI 兼容**：`{"input","voice"?,"response_format"?}` → audio |
| `POST` | `/vc` | 语音转换（把其他音频转成目标音色） |

> 对应桌宠侧配置：⚙️ → TTS → 模式选 `server`，服务地址填 `http://<本机或服务器IP>:8001`。

### 调用示例

```bash
# 健康检查
curl http://127.0.0.1:8001/health

# 合成一句（OpenAI 兼容接口）
curl -X POST http://127.0.0.1:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"こんにちは、今日も一緒に頑張ろうね","voice":"jp","response_format":"wav"}' \
  --output out.wav
```

## 快速开始

### 1. 准备环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

还需要准备：

- **CosyVoice 仓库**（含 `matcha` / `hifigan` 等子模块）
- **基座模型权重**（CosyVoice3）
- **音色 LoRA 权重**（你自己训练的，或用默认音色）
- **ffmpeg**（用于音频格式转换）

### 2. 配置路径

所有路径都可以用环境变量覆盖（见 `config.example.env`）：

```bash
VOICE_MODEL_DIR=/path/to/CosyVoice3-0.5B
VOICE_LORA_JP=/path/to/lora_jp
VOICE_VOCAL_DIR=/path/to/ref_vocals
FFMPEG=/usr/bin/ffmpeg
```

### 3. 启动

```bash
python tts_server.py --port 8001 --gpu-idle-seconds 60
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--port` | 监听端口，默认 8001 |
| `--gpu-idle-seconds` | 空闲多少秒后模型退回 CPU（把显存还给其他程序） |
| `--device` | 初始加载设备，`cpu` 或 `cuda` |

### 4. 接入桌宠

在桌宠里打开 ⚙️ → TTS：

| 配置项 | 填什么 |
|---|---|
| 模式 | `server（常驻服务）` |
| 服务地址 | `http://127.0.0.1:8001`（或部署机器的 IP） |

桌宠会调用 `/v1/audio/speech`，拿到音频后自动播放并驱动口型。

## 资源占用（实测）

| 项目 | 占用 |
|---|---|
| 内存 | ≈ 6.0 GB 常驻 |
| 显存 | ≈ 3.7 GB（仅合成时，空闲后释放） |

> 想省内存：可参考 `tts_server.py` 里的中文支持移除方式（省约 1.9 GB），只保留你需要的语言。

## 目录说明

| 文件 | 说明 |
|---|---|
| `tts_server.py` | **主服务**（HTTP 服务 + 模型常驻 + GPU 调度） |
| `watch_tts_server.py` | 看门狗：服务挂了自动重启 |
| `ja_katakana.py` | 日语片假名处理工具 |
| `guard_threshold.json` | 输出质量护栏阈值（能量/削波检测） |
| `config.example.env` | 环境变量配置模板 |
| `docs/` | 部署与调优文档 |

## 部署

- **Windows 本机**：直接跑 `tts_server.py`，桌宠填 `127.0.0.1`
- **局域网/服务器**：部署到有 GPU 的机器，桌宠填其 IP（如 `http://192.168.1.10:8001`）
- **容器/PVE**：见 `docs/` 下的部署记录

> 模型权重体积大（数 GB）且涉及音色版权，**不随仓库分发**，需自行准备。

## 相关项目

- 🐱 **[桌面伴侣 · Desktop Companion](https://github.com/crsm5192/desktop-companion)** —— 使用本服务的桌宠客户端。
  直接下载：**[便携版](https://github.com/crsm5192/desktop-companion/releases/latest/download/DesktopCompanion.0.1.0.exe)** ｜
  **[安装版](https://github.com/crsm5192/desktop-companion/releases/latest/download/DesktopCompanion.Setup.0.1.0.exe)** ｜
  **[Releases](https://github.com/crsm5192/desktop-companion/releases)**
- 🔊 **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** —— 底层语音合成模型

## 许可

MIT（本仓库代码）。CosyVoice 及其模型权重遵循其各自的原始许可，请自行遵守。
