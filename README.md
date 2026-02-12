[English](#comfyui-llm-chat-node) | [中文](#comfyui-llm-聊天节点)

# ComfyUI LLM Chat Node

This custom node provides a single `LLM Chat` node for any OpenAI-compatible API endpoint. It includes model list caching, optional image input for vision models, retries with fail-word filtering, and deterministic or randomized seeds.

## Features

- **OpenAI-Compatible Only**: Works with any OpenAI-style `/chat/completions` API.
- **Model List Cache**: Fetch models from `/models` and cache by `base_url` with a quick refresh toggle.
- **Multiline Prompts**: System and user prompts support multiline input.
- **Image Support**: Optional image input for vision-language models.
- **Retries + Fail Words**: Auto-retry and re-run if response contains any fail words.
- **Result Cache**: Caches results by inputs (ignores seed) with optional forced rerun.
- **Seed Controls**: Fixed seed or random seed on each run.
- **Three Outputs**: `thinking`, `response`, and `raw_json`.
- **Streaming via httpx**: Uses streaming to reduce long-timeout issues.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/PaddyDu/llm-chat.git
    ```
3.  Install dependencies:
    ```bash
    pip install httpx numpy pillow
    ```
4.  Restart ComfyUI.

## Usage

After installation, add the node to your workflow:
- Right-click on the canvas -> Add Node -> Utils -> AI -> **LLM Chat**

### Configuring the Node

1.  **Connection**:
    - `base_url`: Your API endpoint, for example `https://api.openai.com/v1`.
    - `api_key`: Your API key for the provider.
    - `model`: Select from the cached model list.

2.  **Model List**:
    - `refresh_models`: Fetch and refresh the model list from `/models`.

3.  **Behavior**:
    - `timeout`: Request timeout in seconds.
    - `max_retries`: Retry count for network/API failures.
    - `fail_words`: Comma-separated words that trigger a retry when found in the response.
    - `always_rerun`: Bypass cache and force a fresh API call.
    - `use_random_seed` + `seed`: Randomize or fix the seed value.
    - `incognito`: Passes an `incognito` flag to providers that support it.

4.  **Prompts**:
    - `system_prompt` and `user_prompt`: Multiline prompt inputs.
    - `image` (Optional): Connect an image for vision models.

Outputs:
- `thinking`: Extracted from `<think>...</think>` when present.
- `response`: Final text response.
- `raw_json`: Raw API response.

Notes:
- Model caches are stored under `llm_chat_cache/` per `base_url`.
- The last used `base_url` is remembered for convenience.

---

# ComfyUI LLM 聊天节点

这是一个只面向 OpenAI 兼容 API 的 `LLM Chat` 自定义节点，支持模型列表缓存、可选图像输入、失败词重试以及固定或随机种子。

## 功能特性

- **仅 OpenAI 兼容接口**: 适用于任意 OpenAI 风格的 `/chat/completions` 接口。
- **模型列表缓存**: 通过 `/models` 获取模型列表并按 `base_url` 缓存。
- **多行输入**: 系统提示词与用户提示词支持多行。
- **图像支持**: 可选图像输入，适配视觉模型。
- **重试 + 失败词过滤**: 出现失败词会自动重试。
- **结果缓存**: 基于输入缓存结果（忽略种子），可强制每次重新调用。
- **种子控制**: 支持固定或随机种子。
- **三路输出**: `thinking`、`response`、`raw_json`。
- **httpx 流式请求**: 降低长时间超时问题。

## 安装方法

1.  进入 ComfyUI 的 `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆仓库：
    ```bash
    git clone https://github.com/PaddyDu/llm-chat.git
    ```
3.  安装依赖：
    ```bash
    pip install httpx numpy pillow
    ```
4.  重启 ComfyUI。

## 使用方法

安装后添加节点：
- 画布右键 -> Add Node -> Utils -> AI -> **LLM Chat**

### 配置节点

1.  **连接信息**:
    - `base_url`: API 地址，例如 `https://api.openai.com/v1`。
    - `api_key`: 你的 API Key。
    - `model`: 从缓存的模型列表中选择。

2.  **模型列表**:
    - `refresh_models`: 从 `/models` 重新获取模型列表。

3.  **行为设置**:
    - `timeout`: 超时时间（秒）。
    - `max_retries`: 失败重试次数。
    - `fail_words`: 逗号分隔的失败词，命中即重试。
    - `always_rerun`: 跳过缓存，强制重新调用。
    - `use_random_seed` + `seed`: 随机或固定种子。
    - `incognito`: 将 `incognito` 标记传递给支持该参数的服务端。

4.  **提示词**:
    - `system_prompt` 与 `user_prompt`: 支持多行。
    - `image` (可选): 连接图像输入用于视觉模型。

输出说明：
- `thinking`: 若返回中包含 `<think>...</think>` 会提取到此输出。
- `response`: 最终文本输出。
- `raw_json`: 原始响应内容。

备注：
- 模型缓存保存在 `llm_chat_cache/`，按 `base_url` 分开。
- 会记住上一次使用的 `base_url` 以便快速复用。
