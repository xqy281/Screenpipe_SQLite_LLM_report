# Screenpipe 智能分析与内容生成引擎

**一个支持多模态输入、可迭代工作流的LLM应用框架**

本工具将您的 Screenpipe 本地 OCR 数据转化为一个强大的“数据插座”，通过可定制的提示词模板和先进的多模态能力，将原始的操作记录接入大语言模型，实现从专业工作日报、操作教程到结合流程图的SOP（标准作业程序）等多样化、高质量的内容创作。

它不仅仅是一个脚本，更是一个支持**调试、迭代和优化**的完整内容生成工作流。

## 核心功能

-   **直连本地数据**: 直接安全地连接本地 Screenpipe SQLite 数据库，自动处理**时区转换**，确保数据拉取的准确性。
-   **智能数据清洗**: 结合文本相似度（`sentence-transformers`）和差异对比（`difflib`），有效去除冗余和无意义的屏幕记录，提取高价值信息。
-   **工业级分词与分段策略**:
    -   **模型专属精确计算**: 自动为 Gemini 调用官方 API、为 DeepSeek 加载本地 Tokenizer 进行**精确 Token 计算**，告别估算误差。
    -   **极致性能算法**: 采用**“预计算+指针累加”**策略进行快速粗分，结合**“动态比例移除”**策略进行高效精修，将数十分钟的分段耗时优化至数秒。
-   **多模态输入**: 支持通过命令行传入**图片附件**（如业务流程图），在使用 Gemini 等多模态模型时，实现图文结合的深度分析。
-   **高度可配置的任务系统**:
    -   通过 `config.json` 定义不同的**任务模板（Prompt Templates）**。
    -   每个模板包含独立的**系统提示词**、**分段摘要提示词**和**最终报告提示词**。
    -   通过简单的命令行参数 `--task` 即可切换不同的生成任务。
-   **会话式工作流与二次处理**:
    -   **独立会话存储**: 每一次运行都会创建一个唯一的会话文件夹，存放当次生成的所有分段摘要和最终报告，便于归档和追踪。
    -   **迭代式微调**: 支持通过 `--use_summaries_from` 参数加载之前生成的摘要文件，允许您在**手动修改摘要**后，跳过所有耗时步骤，以极低的成本快速重新生成最终报告。
-   **灵活的参数调整**: 支持通过命令行临时覆盖模型（`--llm`）、温度（`--temperature`）等关键参数，方便调试和实验。
-   **健壮的工程实践**: 内置时间戳日志、模型专属的API调用延迟、全局代理设置等，确保程序运行稳定、可观测。

## 工作流示意图

本工具支持两种核心工作模式：

```
[ 模式一: 完整运行 (From Scratch) ]
                                                     +-------------------------+
OCR数据 --> 清洗 --> 高效分段 --> [摘要1.txt, 摘要2.txt, ...] --> | 最终报告.md (主文件)    |
   |         |        |              | (保存到会话文件夹)         | +-------------------------+
 (耗时)    (较快)   (极快)                                     (保存到会话文件夹)

[ 模式二: 微调运行 (Fine-tuning) ]
                                [摘要1.txt, 摘要2.txt, ...] --> +-------------------------+
                                   (从文件夹加载, 可手动修改)      | 最终报告.md (新版本)    |
                                                                +-------------------------+
                                                                (快速生成并保存到新文件夹)
```

## 安装与配置

### 步骤 1: 准备项目文件

将项目文件（`main.py`, `config.json`, `requirements.txt`等）保存在您的本地计算机上。

### 步骤 2: 准备 DeepSeek Tokenizer (如果使用)

1.  在项目根目录创建一个名为 `deepseek_v2_tokenizer` 的文件夹。
2.  将 DeepSeek 官方提供的 Tokenizer 所有相关文件（`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` 等）放入此文件夹中。

### 步骤 3: 安装依赖

在项目根目录的终端中，运行以下命令：

```bash
pip install -r requirements.txt
```

### 步骤 4: 配置 `config.json`

打开 `config.json` 文件，根据您的需求进行配置。

-   **必须修改的项**:
    -   `llm_config.gemini.api_key`: 填入您的 Google Gemini API 密钥。
    -   `llm_config.deepseek.api_key`: 填入您的 DeepSeek API 密钥。
-   **建议检查的项**:
    -   `llm_config.*.api_call_delay_seconds`: 为有速率限制的模型（如Gemini）设置一个合适的延迟（秒），对于无限制的模型（如DeepSeek）设为 `0`。
    -   `prompt_templates`: 您可以修改已有的提示词，或仿照现有格式添加您自己的任务模板。

## 使用方法

### 模式一：完整运行

从 Screenpipe 数据库拉取数据，完整执行所有步骤。

-   **生成一份工作日报**:
    ```bash
    python main.py --task daily_report --llm gemini
    ```

-   **结合流程图生成一份SOP文档**:
    ```bash
    python main.py --task git_release_sop_generator --llm gemini --attachment "path/to/your/业务流程图.png"
    ```

-   **一个复杂的组合命令示例**:
    ```bash
    python main.py --start_time "2025-06-20T09:00:00" --end_time "2025-06-20T18:00:00" --task tutorial_generator --llm deepseek --temperature 0.2
    ```

### 模式二：微调运行 (从已有摘要生成)

当您对一次“完整运行”生成的最终报告不满意时，可以使用此模式进行快速迭代。

1.  **找到会话文件夹**: 在 `reports/` 目录下找到您想微调的那次运行的文件夹，例如 `reports/2025-06-20_110000_daily_report_gemini/`。
2.  **手动修改摘要**: 进入其下的 `summaries/` 文件夹，用文本编辑器打开并修改一个或多个 `_summary.txt` 文件。
3.  **执行微调命令**:
    ```bash
    python main.py --task daily_report --llm gemini --use_summaries_from "reports/2025-06-20_110000_daily_report_gemini/summaries"
    ```    程序将跳过所有耗时步骤，直接使用您修改后的摘要来生成一份新的最终报告。

## 如何扩展

### 添加一个新的任务模板

1.  打开 `config.json` 文件。
2.  在 `prompt_templates` 对象中，添加一个新的 `key` (例如 `"code_review_assistant"`)。
3.  为这个 `key` 添加一个包含 `description`, `system_prompt`, `chunk_summary_prompt`, 和 `final_report_prompt` 的对象。
4.  保存文件后，您就可以通过 `--task code_review_assistant` 来调用这个新任务了。

## 核心技术亮点

-   **分段算法**: 采用了“预计算+指针累加”的粗分策略，将循环中的字符串操作和重复计算降至最低。结合“动态比例移除”的精修策略，将API调用次数优化至理论最小值，实现了极致的运行效率。
-   **多模态处理**: 通过 `Pillow` 库和对 `google-generativeai` SDK的封装，实现了文本与图像的混合输入，解锁了更深层次的分析能力。
-   **时区自动校正**: 在程序入口处对所有输入时间进行 UTC 转换，从根本上解决了本地时间与数据库存储时间不一致的问题，保证了数据拉取的准确性。