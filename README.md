# Screenpipe 智能分析与内容生成引擎

**一个支持多模态输入、可迭代工作流、并已适配最新 `google-genai` 库的LLM应用框架**

本工具将您的 Screenpipe 本地 OCR 数据转化为一个强大的“数据插座”，通过可定制的提示词模板和先进的语言模型能力（包括 Gemini 的“思考模式”），将原始的操作记录转化为专业工作日报、操作教程、SOP 等多样化、高质量的内容。

它不仅仅是一个脚本，更是一个支持**调试、迭代和优化**的完整内容生成工作流。

## 核心功能

-   **直连本地数据**: 直接安全地连接本地 Screenpipe SQLite 数据库，自动处理**时区转换**，确保数据拉取的准确性。
-   **智能数据清洗**: 结合文本相似度（`sentence-transformers`）和差异对比（`difflib`），有效去除冗余和无意义的屏幕记录，提取高价值信息。
-   **工业级分词与分段策略**:
    -   **模型专属精确计算**: 自动为 Gemini 调用官方 API (`google-genai`)、为 DeepSeek 加载本地 Tokenizer 进行**精确 Token 计算**，告别估算误差。
    -   **极致性能算法**: 采用**“预计算+指针累加”**策略进行快速粗分，结合**“动态比例移除”**策略进行高效精修，将数十分钟的分段耗时优化至数秒。
-   **先进的 Gemini 模型支持**:
    -   **完全适配 `google-genai`**: 已从旧版库完全迁移至最新的 `google-genai` SDK，确保最佳性能和功能兼容性。
    -   **支持思考模式 (Thinking Mode)**: 可在配置文件中为兼容的模型（如 `gemini-2.5-flash-lite`）开启“思考模式”，让模型在生成复杂内容前进行更深度的思考。
    -   **多模态输入**: 支持通过命令行传入**图片附件**（如业务流程图），实现图文结合的深度分析。
-   **高度可配置的任务系统**:
    -   通过 `config.json` 定义不同的**任务模板（Prompt Templates）**。
    -   每个模板包含独立的**系统提示词**、**分段摘要提示词**和**最终报告提示词**。
    -   通过简单的命令行参数 `--task` 即可切换不同的生成任务。
-   **会话式工作流与二次处理**:
    -   **独立会话存储**: 每一次运行都会创建一个唯一的会话文件夹，存放当次生成的所有中间文件（如清洗后的数据、分段摘要）和最终报告，便于归档和追踪。
    -   **迭代式微调**: 支持通过 `--use_summaries_from` 或 `--load_cleaned_data` 参数加载之前会话的产物，允许您在**手动修改摘要**或**复用清洗数据**后，跳过所有耗时步骤，以极低的成本快速重新生成最终报告。
-   **灵活的参数调整与调试**:
    -   支持通过命令行临时覆盖模型（`--llm`）、温度（`--temperature`）等关键参数。
    -   提供 `--debug` 模式，可将**每一次**发送给 LLM 的完整 Prompt 保存为快照文件，是诊断和优化提示词的强大工具。

## 工作流示意图

本工具支持多种灵活的工作模式：

```
[ 模式一: 完整运行 (From Scratch) ]
                                                     +-------------------------+
OCR数据 --> 清洗 --> 高效分段 --> [摘要1.txt, 摘要2.txt, ...] --> | 最终报告.md (主文件)    |
   |         |        |              | (保存到会话文件夹)         | +-------------------------+
 (耗时)    (较快)   (极快)                                     (保存到会话文件夹)

[ 模式二: 从已清洗数据运行 ]
                                                     +-------------------------+
cleaned_data.json --> 高效分段 --> [摘要1.txt, 摘要2.txt, ...] --> | 最终报告.md (新版本)    |
(从文件夹加载)                                                    +-------------------------+

[ 模式三: 从摘要微调运行 ]
                                [摘要1.txt, 摘要2.txt, ...] --> +-------------------------+
                                   (从文件夹加载, 可手动修改)      | 最终报告.md (新版本)    |
                                                                +-------------------------+
                                                                (快速生成并保存到新文件夹)
```

## 安装与配置

### 步骤 1: 准备项目文件

将项目文件（`main.py`, `config.json`, `requirements.txt`等）保存在您的本地计算机上。

### 步骤 2: 安装/更新 Google GenAI 库 (重要)

为了使用 Gemini 的最新功能（如思考模式），必须使用新的 `google-genai` 库。请在终端中执行以下命令：

```bash
# 卸载可能存在的旧库，防止冲突
pip uninstall google-generativeai -y

# 安装新的、正确的库
pip install google-genai
```

### 步骤 3: 准备 DeepSeek Tokenizer (如果使用)

1.  在项目根目录创建一个名为 `deepseek_v2_tokenizer` 的文件夹。
2.  将 DeepSeek 官方提供的 Tokenizer 所有相关文件（`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` 等）放入此文件夹中。

### 步骤 4: 安装其他依赖

在项目根目录的终端中，运行以下命令：

```bash
pip install -r requirements.txt
```

### 步骤 5: 配置 `config.json`

打开 `config.json` 文件，根据您的需求进行配置。

-   **必须修改的项**:
    -   `llm_config.gemini.api_key`: 填入您的 Google Gemini API 密钥。
    -   `llm_config.deepseek.api_key`: 填入您的 DeepSeek API 密钥。
-   **建议检查的项**:
    -   `llm_config.gemini.enable_thinking_mode`: 如果你想为 Gemini 启用思考模式，请确保此项为 `true`。
    -   `llm_config.*.api_call_delay_seconds`: 为有速率限制的模型（如Gemini）设置一个合适的延迟（秒），对于无限制的模型（如DeepSeek）设为 `0`。
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

-   **使用 `gemini` 并启用思考模式** (需在 `config.json` 中配置 `enable_thinking_mode: true`):
    ```bash
    python main.py --llm gemini
    ```

### 高级用法

-   **指定时间范围生成教程**:
    ```bash
    python main.py --llm gemini --task tutorial_generator --start_time "2025-06-20T09:00:00" --end_time "2025-06-20T18:00:00"
    ```

-   **附加图片生成带图文解释的报告**:
    ```bash
    python main.py --llm gemini --attachment "path/to/your/flowchart.png"
    ```

-   **启用调试模式进行 Prompt 诊断**:
    ```bash
    python main.py --task your_task --debug
    ```
    这会在当次运行的会话文件夹下，创建一个 `prompt_snapshots` 子目录，并将**每一次**发送给 LLM 的、包含完整内容的 Prompt 保存为 `.txt` 文件。

### 迭代式工作流

当您对一次“完整运行”生成的最终报告不满意时，可以使用此模式进行快速迭代。

1.  **找到会话文件夹**: 在 `reports/` 目录下找到您想微调的那次运行的文件夹，例如 `reports/2025-06-30_..._daily_report_gemini/`。
2.  **选择迭代方式**:
    *   **方式A：修改摘要后重新生成**：进入其下的 `summaries/` 文件夹，用文本编辑器打开并修改一个或多个 `_summary.txt` 文件。然后执行：
        ```bash
        python main.py --task daily_report --use_summaries_from "reports/2025-06-30_..._daily_report_gemini/summaries"
        ```
    *   **方式B：使用已清洗数据重新处理**：如果您认为问题出在分段或摘要阶段，而不是数据清洗，可以复用 `cleaned_data.json` 文件。
        ```bash
        python main.py --task daily_report --load_cleaned_data "reports/2025-06-30_..._daily_report_gemini/cleaned_data.json"
        ```
程序将跳过所有前面的耗时步骤，直接从您指定的阶段开始，快速生成一份新的最终报告。

## 如何扩展

### 添加一个新的任务模板

1.  打开 `config.json` 文件。
2.  在 `prompt_templates` 对象中，添加一个新的 `key` (例如 `"code_review_assistant"`)。
3.  为这个 `key` 添加一个包含 `description`, `system_prompt`, `chunk_summary_prompt`, 和 `final_report_prompt` 的对象。
4.  保存文件后，您就可以通过 `--task code_review_assistant` 来调用这个新任务了。

### 添加新的大语言模型

如果一个新的模型与 OpenAI API 兼容，您只需在 `config.json` 的 `llm_config` 中添加一个新的配置项，并可考虑在 `LLMConnectorFactory` 中添加相应的逻辑即可。

## 注意事项

-   本工具依赖于本地已安装并正常运行的 **Screenpipe** 应用。
-   使用大语言模型会产生 API 调用费用，请关注您的账户用量。
-   请务必遵循**步骤2**的指引，正确安装 `google-genai` 库，以确保所有功能正常。