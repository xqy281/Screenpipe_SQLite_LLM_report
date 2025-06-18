# Screenpipe 智能分析与内容生成引擎

一个灵活、可配置的工具，能将您的 Screenpipe 本地 OCR 数据转化为结构化的内容，如工作日报、操作教程等。它作为一个“底层数据插座”，通过可定制的提示词模板，将原始的操作记录接入强大的大语言模型，实现多样化的内容创作。

## 核心功能

-   **直连本地数据**: 直接安全地连接本地 Screenpipe SQLite 数据库，无需通过网络或第三方服务。
-   **智能数据清洗**: 结合文本相似度（`sentence-transformers`）和差异对比（`difflib`），有效去除冗余和无意义的屏幕记录，提取高价值信息。
-   **精准分段策略**:
    -   支持 **本地精确分词**（已集成 DeepSeek Tokenizer），确保发送给 API 的数据块（Chunk）大小不超出模型上下文限制。
    -   对于其他模型，采用**安全系数**策略，保证处理超长文本时的稳定性。
-   **多模型支持**: 内置对 Google Gemini 和 OpenAI 兼容 API（如 DeepSeek）的支持，并可通过配置文件轻松扩展。
-   **高度可配置的任务系统**:
    -   通过 `config.json` 定义不同的**任务模板（Prompt Templates）**。
    -   每个模板包含独立的**系统提示词**、**分段摘要提示词**和**最终报告提示词**。
    -   通过简单的命令行参数 `--task` 即可切换不同的生成任务（如 `daily_report` vs `tutorial_generator`）。
-   **灵活的参数调整**: 支持通过命令行临时覆盖模型（`--llm`）和温度（`--temperature`）等关键参数，方便调试和实验。
-   **动态文件名**: 生成的报告文件名包含时间戳、任务名和模型名，方便归档和溯源。

## 项目结构

```
your_project_folder/
├── deepseek_v2_tokenizer/      <-- DeepSeek 本地 Tokenizer 文件
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
│
├── main.py                     <-- 主程序脚本
├── config.json                 <-- 核心配置文件
├── requirements.txt            <-- Python 依赖
└── reports/                    <-- 生成报告的输出目录
```

## 安装与配置

### 步骤 1: 克隆或下载项目

将项目文件保存在您的本地计算机上。

### 步骤 2: 准备 DeepSeek Tokenizer (如果使用)

> **重要**: 这是确保对 DeepSeek 模型进行精确 Token 计算的关键步骤。

1.  在项目根目录创建一个名为 `deepseek_v2_tokenizer` 的文件夹。
2.  将 DeepSeek 官方提供的 Tokenizer 所有相关文件（`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` 等）放入此文件夹中。

### 步骤 3: 安装依赖

在项目根目录的终端中，运行以下命令来安装所有必需的 Python 库：

```bash
pip install -r requirements.txt
```

### 步骤 4: 配置 `config.json`

打开 `config.json` 文件，根据您的需求进行配置。

-   **必须修改的项**:
    -   `llm_config.gemini.api_key`: 填入您的 Google Gemini API 密钥。
    -   `llm_config.deepseek.api_key`: 填入您的 DeepSeek API 密钥。

-   **建议检查的项**:
    -   `llm_config.gemini.proxy`: 如果您在中国大陆或其他需要代理的地区使用 Gemini，请确保代理地址正确。如果不需要，可以设为 `null` 或删除此行。
    -   `prompt_templates`: 您可以修改已有的提示词，或仿照现有格式添加您自己的任务模板。

## 使用方法

所有操作都在项目根目录的终端中执行。

### 基本用法

-   **生成一份默认的工作日报** (使用配置文件中默认的 `llm_provider`):
    ```bash
    python main.py
    ```

-   **指定使用 `deepseek` 模型生成日报**:
    ```bash
    python main.py --llm deepseek
    ```

### 高级用法

-   **生成一份教程**:
    通过 `--task` 参数选择在 `config.json` 中定义的 `tutorial_generator` 任务。
    ```bash
    python main.py --llm gemini --task tutorial_generator
    ```

-   **指定时间范围**:
    使用 `--start_time` 和 `--end_time` 参数来处理特定时间段的数据。
    ```bash
    python main.py --start_time "2025-06-18T09:00:00" --end_time "2025-06-18T18:00:00"
    ```

-   **临时调整 Temperature**:
    使用 `--temperature` 参数覆盖配置文件中的设置，以获得不同风格的输出（值越低越确定，越高越有创意）。
    ```bash
    python main.py --llm deepseek --temperature 0.2
    ```

-   **组合使用**:
    ```bash
    python main.py --start_time "2025-06-10T00:00:00" --end_time "2025-06-10T23:59:59" --llm deepseek --task tutorial_generator --temperature 0.5
    ```

## 如何扩展

### 添加一个新的任务模板

1.  打开 `config.json` 文件。
2.  在 `prompt_templates` 对象中，添加一个新的 `key` (例如 `"meeting_summary"`)。
3.  为这个 `key` 添加一个包含 `description`, `system_prompt`, `chunk_summary_prompt`, 和 `final_report_prompt` 的对象。

**示例**: 添加一个“会议纪要”任务
```json
{
  ...
  "prompt_templates": {
    "daily_report": { ... },
    "tutorial_generator": { ... },
    "meeting_summary": {
      "description": "根据会议期间的屏幕操作和少量文本，生成一份会议纪要。",
      "system_prompt": "你是一名专业的会议记录员，擅长从零散的信息中整理出条理清晰的会议纪要。",
      "chunk_summary_prompt": "请总结以下会议期间的操作记录，提炼出讨论的要点和展示的关键内容：\n\n{chunk_text}",
      "final_report_prompt": "请根据以下所有信息，生成一份完整的会议纪要。纪要应包含【会议主题】、【参会人员】、【主要议题】和【行动项】：\n\n{all_summaries}"
    }
  },
  ...
}
```
4.  保存文件后，您就可以通过以下命令来使用这个新任务了：
    ```bash
    python main.py --task meeting_summary
    ```

### 添加新的大语言模型

如果一个新的模型与 OpenAI API 兼容，您只需在 `config.json` 的 `llm_config` 中添加一个新的配置项即可。

## 注意事项

-   本工具依赖于本地已安装并正常运行的 **Screenpipe** 应用。
-   使用大语言模型会产生 API 调用费用，请关注您的账户用量。
-   在 Windows 系统上首次运行 `sentence-transformers` 时，可能会出现关于 `symlinks` 的警告，这通常不影响程序功能。