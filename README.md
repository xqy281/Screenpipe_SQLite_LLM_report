# 生产级 Screenpipe 智能分析与内容生成引擎

**一个具备高并发速率控制、三步智能提纯与零成本演习模式的LLM应用框架**

本工具将您的 Screenpipe 本地 OCR 数据转化为一个强大的“数据插座”，通过可定制的提示词模板和先进的语言模型能力，将原始、混乱的操作记录转化为专业工作日报、操作教程、SOP 等多样化、高质量的内容。

它不仅仅是一个脚本，更是一个支持**高并发请求、智能容错、迭代优化和零成本调试**的完整内容生成工作流，专为解决真实世界中的复杂 API 交互而设计。

## 核心功能

-   **直连本地数据**: 直接安全地连接本地 Screenpipe SQLite 数据库，自动处理**时区转换**，确保数据拉取的准确性。

-   **智能数据清洗**: 结合文本相似度（`sentence-transformers`）和差异对比（`difflib`），有效去除冗余和无意义的屏幕记录，提取高价值信息。

-   **工业级并发与速率控制 (`RateControlManager`)**:
    -   **多维度精准控制**: 内置一个全局的、线程安全的速率控制中心，同时管理 **RPM (每分钟请求数)**、**TPM (每分钟Token数)** 和**并发数**，确保在高并发场景下绝不超限。
    -   **智能全局冷却**: 当收到 API 返回的 `429` 速率超限错误时，能自动解析服务器建议的 `retryDelay`，并启动一个**全局“紧急刹车”**，强制所有线程暂停，避免“惊群效应”导致连锁失败。
    -   **健壮的指数退避**: 对 `500`, `502`, `503` 等服务器瞬时错误，以及底层的 `SSL/EOF` 网络错误，都能自动执行**指数退避重试**，极大提升了长时间运行的稳定性。
    -   **高度可配置**: 所有速率限制参数（RPM, TPM, Concurrency）均可在 `config.json` 中轻松配置，以适应不同等级的 API Key。

-   **两种核心处理流程**:
    1.  **标准分段摘要**: 采用“预计算+指针累加”的高效分段算法，将大量文本数据切分为适合 LLM 上下文窗口的块，并逐一总结。
    2.  **🧪 LLM 智能提纯 (`--enable-llm-refinement`)**: 针对专业领域（如编程）的实验性三步流程，以生成更具叙事性和逻辑性的报告：
        -   **① 原子化并发分类**: 使用 LLM 为每一条 OCR 记录高并发地打上活动标签（如 `EmbeddedCoding`, `BuildAndCompile`）。
        -   **② 编程分组**: 在本地根据连续的活动标签，将记录聚合成有意义的“工作阶段”。
        -   **③ 上下文感知总结**: 再次使用 LLM，为每一个“工作阶段”生成一个高度浓缩、人类可读的摘要。

-   **零成本演习与调试 (`--dry-run`)**:
    -   **仅生成，不付费**: 激活此模式后，程序会完整执行所有数据处理流程，并生成全部的 `prompt_snapshots` 文件，但**会跳过所有付费的 `generate_content` API 调用**。
    -   **精确分块模拟**: 在演习模式下，程序**仍会调用免费的 `count_tokens` API**，以确保数据分块的结果与真实运行时完全一致。
    -   **快速验证**: 是调试提示词、验证数据处理逻辑的绝佳工具，完全无需担心 API 费用。

-   **会话式与迭代式工作流**:
    -   **独立会话存储**: 每一次运行都会创建一个唯一的会话文件夹，存放当次的所有中间产物（清洗数据、分段摘要、Prompt快照）和最终报告。
    -   **迭代式微调**: 支持通过 `--use_summaries_from` 或 `--load_cleaned_data` 参数加载之前会话的产物，允许您在**手动修改摘要**或**调整终版报告提示词**后，以极低的成本快速重新生成最终报告。

## 工作流示意图

本工具支持多种灵活的工作模式，以适应不同需求：

```
[ 模式一: 完整运行 (标准流程) ]
                                                     +-------------------------+
OCR数据 --> 清洗 --> 高效分段 --> [摘要1.txt, 摘要2.txt, ...] --> | 最终报告.md (主文件)    |
   |         |        |              | (保存到会话文件夹)         | +-------------------------+
 (耗时)    (较快)   (极快)                                     (保存到会话文件夹)

[ 模式二: 完整运行 (LLM提纯流程) ]
                                                     +-------------------------+
OCR数据 --> 清洗 --> LLM分类 --> 编程分组 --> LLM总结 --> | 最终报告.md (高质量版)  |
                                                     +-------------------------+

[ 模式三: 从中间产物迭代 ]
                                                     +-------------------------+
cleaned_data.json --> [模式一或二的后续步骤] -->            | 最终报告.md (新版本)    |
(从文件夹加载)                                                    +-------------------------+
                                                     +-------------------------+
[摘要1.txt, 摘要2.txt, ...] --> (跳过所有摘要步骤) -->      | 最终报告.md (新版本)    |
(从文件夹加载, 可手动修改)                                    +-------------------------+

[ 模式四: 零成本演习 (--dry-run) ]
                                                     +-------------------------+
OCR数据 --> [完整流程，但不调用付费API] --> [空摘要.txt, ...] --> | 空报告.md + Prompt快照 |
                                                     +-------------------------+
```

## 安装与配置

### 步骤 1: 准备项目文件

将项目文件（`main.py`, `config.json`, `requirements.txt`等）保存在您的本地计算机上。

### 步骤 2: 安装依赖

为了确保使用正确的库版本，强烈建议先执行卸载命令，再进行安装。

```bash
# 卸载可能存在的旧库，防止冲突
pip uninstall google-generativeai -y

# 安装新的、正确的库和所有其他依赖
pip install -r requirements.txt
```

### 步骤 3: 配置 `config.json`

打开 `config.json` 文件，根据您的需求进行配置。

-   **必须修改的项**:
    -   `llm_config.gemini.api_key`: 填入您的 Google Gemini API 密钥。
    -   `llm_config.deepseek.api_key`: 填入您的 DeepSeek API 密钥。
-   **强烈建议检查的项**:
    -   `llm_config.gemini.rate_limiting`: 根据您 API Key 的等级，配置准确的 `rpm`, `tpm`, 和 `concurrency`。
-   **可定制项**:
    -   `prompt_templates`: 您可以修改已有的提示词，或仿照现有格式添加您自己的任务模板。
    -   `llm_refinement_config`: 为“智能提纯”流程定制分类和总结的提示词。

## 使用方法

所有操作都在项目根目录的终端中执行。

### 基本用法

-   **生成一份默认的工作日报** (使用标准流程和配置文件中的默认LLM):
    ```bash
    python main.py
    ```

-   **指定使用 `deepseek` 模型**:
    ```bash
    python main.py --llm deepseek
    ```

### 高级工作流

-   **零成本演习与Prompt审查 (`--dry-run`)**:
    此命令会完整地运行数据处理和分块，生成所有 Prompt 快照和空的摘要文件，但不会有任何付费 API 调用。
    ```bash
    python main.py --task daily_report --dry-run
    ```

-   **启用LLM智能提纯流程 (`--enable-llm-refinement`)**:
    使用三步提纯法生成更高质量的报告（当前仅`gemini`支持）。
    ```bash
    python main.py --task sean_report --enable-llm-refinement
    ```

-   **从已清洗数据重新运行**:
    当您想用同一份数据，尝试不同的提示词或流程时使用。
    ```bash
    python main.py --task new_report_task --load_cleaned_data "reports/path/to/your/cleaned_data.json"
    ```

-   **从已生成的摘要重新生成最终报告**:
    当您手动修改了部分摘要，或只想调整最终报告的生成逻辑时使用。
    ```bash
    python main.py --task daily_report --use_summaries_from "reports/path/to/your/summaries"
    ```

### 组合用法

-   **指定时间范围、附加图片并启用调试模式**:
    ```bash
    python main.py --task git_release_sop_generator --start_time "2025-07-05T09:00:00" --attachment "path/to/flowchart.png" --debug
    ```

## 如何扩展

### 添加一个新的任务模板

1.  打开 `config.json` 文件。
2.  在 `prompt_templates` 对象中，添加一个新的 `key` (例如 `"code_review_assistant"`)。
3.  为这个 `key` 添加一个包含 `description`, `system_prompt`, `chunk_summary_prompt`, 和 `final_report_prompt` 的对象。
4.  保存文件后，您就可以通过 `--task code_review_assistant` 来调用这个新任务了。

## 注意事项

-   本工具依赖于本地已安装并正常运行的 **Screenpipe** 应用。
-   使用大语言模型会产生 API 调用费用，请关注您的账户用量。
-   请务必遵循**步骤2**的指引，正确安装 `google-genai` 库，以确保所有功能正常。