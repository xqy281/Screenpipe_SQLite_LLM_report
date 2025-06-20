# main.py
import os
import json
import argparse
import difflib
import sqlite3
import platform
import time
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# --- 动态导入检查 ---
try:
    import tiktoken
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai
    from openai import OpenAI
    import transformers
    from PIL import Image
except ImportError as e:
    print(f"依赖库导入失败: {e}")
    print("请确保已通过 'pip install -r requirements.txt' 安装所有依赖。")
    exit(1)


# --- 带时间戳的日志函数 ---
def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# *** 关键变更: 引入面向对象的 LLM 连接器架构 ***


class BaseLLMConnector:
    """所有 LLM 连接器的基类，定义了统一的接口。"""

    def __init__(self, provider_config: Dict[str, Any]):
        self.provider_config = provider_config
        self.model_name = provider_config.get("model_name")
        self.client = self._initialize_client()
        log_with_timestamp(
            f"🤖 {self.__class__.__name__} 已初始化 (模型: {self.model_name})"
        )

    def _initialize_client(self):
        raise NotImplementedError

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


class GeminiConnector(BaseLLMConnector):
    """处理 Google Gemini 模型的连接器。"""

    def _initialize_client(self):
        genai.configure(api_key=self.provider_config["api_key"])
        return genai.GenerativeModel(self.model_name)

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ) -> str:
        log_with_timestamp(f"🚀 发起 Gemini API 调用 (Temperature: {temperature})...")
        try:
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            content = [user_prompt]
            if attachment_data:
                if attachment_type == "image":
                    log_with_timestamp("  - 正在将图片附件添加到 Gemini 请求中...")
                    content.append(attachment_data)
                elif attachment_type == "text":
                    content[0] = (
                        f"### 补充上下文:\n{attachment_data}\n\n### 主要任务:\n{user_prompt}"
                    )
            if system_prompt:
                content[0] = f"{system_prompt}\n\n---\n\n{content[0]}"
            response = self.client.generate_content(
                content, generation_config=generation_config
            )
            return response.text
        except Exception as e:
            log_with_timestamp(f"❌ Gemini API 调用失败: {e}")
            return f"[LLM 调用错误: {e}]"
        return "[LLM 调用返回空]"

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text).total_tokens


class DeepSeekConnector(BaseLLMConnector):
    """处理 DeepSeek 和其他 OpenAI 兼容 API 的连接器。"""

    def _initialize_client(self):
        return OpenAI(
            api_key=self.provider_config["api_key"],
            base_url=self.provider_config["base_url"],
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ) -> str:
        log_with_timestamp(f"🚀 发起 DeepSeek API 调用 (Temperature: {temperature})...")

        # 当前的保护性措施
        if attachment_type == "image":
            log_with_timestamp(
                "❌ 错误: DeepSeek 的 OpenAI 兼容 API 当前不支持直接的图像输入。"
            )
            log_with_timestamp(
                "   (模型本身具备多模态能力，但需要等待官方更新其公共API以支持此功能)"
            )
            return "[错误: 此模型 API 不支持图像输入]"

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if attachment_type == "text" and attachment_data:
                user_prompt = f"### 补充上下文:\n{attachment_data}\n\n### 主要任务:\n{user_prompt}"
            messages.append({"role": "user", "content": user_prompt})
            chat_completion = self.client.chat.completions.create(
                messages=messages, model=self.model_name, temperature=temperature
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            log_with_timestamp(f"❌ DeepSeek API 调用失败: {e}")
            return f"[LLM 调用错误: {e}]"
        return "[LLM 调用返回空]"

    def count_tokens(self, text: str) -> int:
        # DeepSeek 的 Token 计算由 DailyReportGenerator 中的本地 Tokenizer 处理
        # 这个方法在这里只是为了满足接口统一性，实际上不会被调用
        raise NotImplementedError("DeepSeek token counting should be handled locally.")


class LLMConnectorFactory:
    """根据提供商名称创建相应的连接器实例。"""

    @staticmethod
    def create(provider_name: str, provider_config: Dict[str, Any]) -> BaseLLMConnector:
        if provider_name == "gemini":
            return GeminiConnector(provider_config)
        elif provider_name == "deepseek":
            return DeepSeekConnector(provider_config)
        else:
            raise ValueError(f"不支持的 LLM provider: {provider_name}")


# --- 模块 1.1: 数据获取模块 ---
class DataFetcher:
    # ... 此部分代码无变化 ...
    def __init__(self):
        self.db_path = self._get_db_path()
        if not self.db_path or not self.db_path.exists():
            log_with_timestamp(
                f"❌ 错误: 未能找到 Screenpipe 数据库。预期路径: {self.db_path}"
            )
            exit(1)
        log_with_timestamp(f"🔍 成功定位 Screenpipe 数据库: {self.db_path}")

    def _get_db_path(self) -> Path | None:
        system = platform.system()
        home = Path.home()
        if system == "Windows":
            path1 = home / ".screenpipe/db.sqlite"
            path2 = home / "AppData/Roaming/Screenpipe/db.sqlite"
            if path1.exists():
                return path1
            return path2
        if system == "Darwin":
            return home / "Library/Application Support/Screenpipe/db.sqlite"
        if system == "Linux":
            return home / ".config/Screenpipe/db.sqlite"
        return None

    def fetch_data(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        log_with_timestamp(
            f"💾 正在从数据库获取 {start_time.isoformat()} 到 {end_time.isoformat()} 的 OCR 数据..."
        )
        query = "SELECT ocr.text, frm.timestamp AS captured_at FROM frames AS frm JOIN ocr_text AS ocr ON frm.id = ocr.frame_id WHERE frm.timestamp >= ? AND frm.timestamp <= ? ORDER BY frm.timestamp ASC;"
        records = []
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (start_time.isoformat(), end_time.isoformat()))
                for row in cursor.fetchall():
                    records.append(dict(row))
            log_with_timestamp(f"✅ 成功获取 {len(records)} 条 OCR 记录。")
            return records
        except sqlite3.Error as e:
            log_with_timestamp(f"❌ 数据库查询失败: {e}")
            return []


# --- 核心应用类 ---
class DailyReportGenerator:
    def __init__(self, config: Dict[str, Any], cli_args: argparse.Namespace):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.cli_args = cli_args
        self.llm_provider_name = cli_args.llm or self.config["llm_provider"]
        log_with_timestamp(f"🔧 LLM 提供商已确定: {self.llm_provider_name}")
        self.provider_config = self.config["llm_config"][self.llm_provider_name]

        # *** 关键变更: 使用工厂模式创建连接器 ***
        self.llm_connector = LLMConnectorFactory.create(
            provider_name=self.llm_provider_name, provider_config=self.provider_config
        )

        self.task_template = self._load_task_template(cli_args.task)
        if cli_args.temperature is not None:
            self.temperature = cli_args.temperature
            log_with_timestamp(
                f"🔧 已通过命令行参数设置 Temperature: {self.temperature}"
            )
        else:
            self.temperature = self.provider_config.get("temperature", 0.7)
        log_with_timestamp("正在加载 NLP 模型...")
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_tokenizers()
        log_with_timestamp("模型加载完成。")

    def _load_task_template(self, task_name: str) -> Dict[str, str]:
        templates = self.config.get("prompt_templates", {})
        if task_name not in templates:
            log_with_timestamp(f"❌ 错误: 任务 '{task_name}' 在配置文件中未定义。")
            log_with_timestamp(f"可用的任务有: {list(templates.keys())}")
            exit(1)
        log_with_timestamp(
            f"🚀 已选择任务: '{task_name}' - {templates[task_name].get('description')}"
        )
        return templates[task_name]

    def _setup_tokenizers(self):
        self.precise_tokenizer = None
        self.rough_tokenizer = tiktoken.get_encoding("cl100k_base")
        log_with_timestamp("  - 粗算将使用: tiktoken")
        if self.llm_provider_name == "deepseek":
            tokenizer_path = self.provider_config.get("tokenizer_path")
            if tokenizer_path and os.path.isdir(tokenizer_path):
                try:
                    log_with_timestamp(
                        "  - 正在加载 DeepSeek 本地 Tokenizer 用于精算..."
                    )
                    self.precise_tokenizer = transformers.AutoTokenizer.from_pretrained(
                        tokenizer_path, trust_remote_code=True
                    )
                except Exception as e:
                    log_with_timestamp(f"  - ❌ 加载 DeepSeek Tokenizer 失败: {e}。")
        log_with_timestamp(
            f"  - 精算方式已确定: {'local_exact' if self.precise_tokenizer else 'api'}"
        )

    def _clean_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        log_with_timestamp("🧹 开始数据清洗...")
        high_similarity_threshold = self.config["high_similarity_threshold"]
        min_diff_chars = self.config["min_diff_chars"]
        first_valid_index = next(
            (i for i, r in enumerate(records) if r.get("text")), -1
        )
        if first_valid_index == -1:
            return []
        cleaned_records = [records[first_valid_index]]
        for i in range(first_valid_index + 1, len(records)):
            current_record, last_kept_record = records[i], cleaned_records[-1]
            text_current, text_last_kept = current_record.get(
                "text", ""
            ), last_kept_record.get("text", "")
            if not text_current:
                continue
            embeddings = self.similarity_model.encode([text_last_kept, text_current])
            if (
                util.cos_sim(embeddings[0], embeddings[1]).item()
                > high_similarity_threshold
            ):
                continue
            diff = difflib.ndiff(
                text_last_kept.splitlines(keepends=True),
                text_current.splitlines(keepends=True),
            )
            if (
                sum(len(line[2:]) for line in diff if line.startswith("+ "))
                < min_diff_chars
            ):
                continue
            cleaned_records.append(current_record)
        log_with_timestamp(
            f"✅ 清洗完成。记录从 {len(records)} 条减少到 {len(cleaned_records)} 条。"
        )
        return cleaned_records

    def _estimate_precise_tokens(self, text: str) -> int:
        if self.llm_provider_name == "gemini":
            return self.llm_connector.count_tokens(text)
        elif self.llm_provider_name == "deepseek" and self.precise_tokenizer:
            return len(self.precise_tokenizer.encode(text))
        # 如果没有精确方法，回退到粗算
        return self._estimate_rough_tokens(text)

    def _estimate_rough_tokens(self, text: str) -> int:
        return len(self.rough_tokenizer.encode(text))

    def _chunk_data_efficiently(
        self, records: List[Dict[str, Any]], max_chunk_tokens: int
    ) -> List[str]:
        log_with_timestamp("⏳ 正在高效地对数据进行分段...")
        log_with_timestamp("  - 步骤1: 正在对所有记录进行Token粗算预计算...")
        records_with_rough_tokens = [
            (
                record,
                self._estimate_rough_tokens(
                    f"Timestamp: {record['captured_at']}\n{record['text']}"
                ),
            )
            for record in records
        ]
        log_with_timestamp("  - ✅ 预计算完成。")
        chunks = []
        start_index = 0
        while start_index < len(records_with_rough_tokens):
            current_rough_tokens = 0
            end_index = start_index
            while end_index < len(records_with_rough_tokens):
                current_rough_tokens += records_with_rough_tokens[end_index][1]
                if current_rough_tokens > max_chunk_tokens and end_index > start_index:
                    current_rough_tokens -= records_with_rough_tokens[end_index][1]
                    break
                end_index += 1
            current_chunk_records = [
                rec for rec, token in records_with_rough_tokens[start_index:end_index]
            ]
            temp_chunk_text = "\n\n---\n\n".join(
                [
                    f"Timestamp: {r['captured_at']}\n{r['text']}"
                    for r in current_chunk_records
                ]
            )
            log_with_timestamp(
                f"  - 粗算打包完成一个块 (记录 {start_index}-{end_index-1}，粗算 {current_rough_tokens})，开始精算..."
            )
            precise_tokens = self._estimate_precise_tokens(temp_chunk_text)
            log_with_timestamp(f"  - 精算结果: {precise_tokens} tokens。")
            while precise_tokens > max_chunk_tokens and len(current_chunk_records) > 1:
                log_with_timestamp(
                    f"  - 精算后仍超限 (溢出 {precise_tokens - max_chunk_tokens} tokens)，开始动态移除..."
                )
                token_overflow = precise_tokens - max_chunk_tokens
                avg_tokens_per_record = precise_tokens / len(current_chunk_records)
                num_to_remove = (
                    math.ceil(token_overflow / avg_tokens_per_record)
                    if avg_tokens_per_record > 0
                    else 1
                )
                num_to_remove = max(1, num_to_remove)
                log_with_timestamp(
                    f"  - 动态计算：预估需移除 {num_to_remove} 条记录..."
                )
                current_chunk_records = current_chunk_records[:-num_to_remove]
                end_index -= num_to_remove
                temp_chunk_text = "\n\n---\n\n".join(
                    [
                        f"Timestamp: {r['captured_at']}\n{r['text']}"
                        for r in current_chunk_records
                    ]
                )
                precise_tokens = self._estimate_precise_tokens(temp_chunk_text)
                log_with_timestamp(f"  - 修正后精算结果: {precise_tokens} tokens。")
            chunks.append(temp_chunk_text)
            start_index = end_index
        log_with_timestamp(f"✅ 分段完成。数据被分为 {len(chunks)} 个段落。")
        return chunks

    def _generate_final_report(
        self,
        llm_context: str,
        run_output_dir: Path,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        log_with_timestamp("🖋️ 开始生成最终报告...")
        final_prompt_template = self.task_template["final_report_prompt"]
        final_prompt = final_prompt_template.format(all_summaries=llm_context)
        final_report = self.llm_connector.generate(
            user_prompt=final_prompt,
            system_prompt=self.task_template.get("system_prompt"),
            temperature=self.temperature,
            attachment_data=attachment_data,
            attachment_type=attachment_type,
        )
        report_path = run_output_dir / "final_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        log_with_timestamp(f"\n🎉 成功！报告已保存至: {report_path}")

    def run(
        self,
        start_time: datetime,
        end_time: datetime,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        run_output_dir = (
            Path(self.config["output_path"])
            / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{self.cli_args.task}_{self.llm_provider_name}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_with_timestamp(f"📂 本次运行会话目录已创建: {run_output_dir}")
        raw_records = self.data_fetcher.fetch_data(start_time, end_time)
        cleaned_records = self._clean_data(raw_records)
        if not cleaned_records:
            log_with_timestamp("ℹ️ 清洗后无有效数据，无法生成报告。")
            return
        log_with_timestamp("📊 正在使用精确方法计算总 Token 数，请稍候...")
        full_text = "\n\n---\n\n".join(
            [f"Timestamp: {r['captured_at']}\n{r['text']}" for r in cleaned_records]
        )
        total_tokens = self._estimate_precise_tokens(full_text)
        log_with_timestamp(f"📊 精确总 Token 数计算完成: {total_tokens}")
        llm_context = ""
        llm_context_window = self.provider_config["context_window"]
        effective_direct_summary_threshold = min(
            self.config["direct_summary_threshold"], llm_context_window
        )
        if total_tokens < effective_direct_summary_threshold:
            log_with_timestamp("📈 Token 总数小于阈值，直接生成最终报告。")
            llm_context = full_text
        else:
            log_with_timestamp("📉 Token 总数超过阈值，执行分段摘要流程。")
            max_chunk_tokens = (
                self.provider_config["context_window"] - self.config["token_headroom"]
            )
            log_with_timestamp(f"  - 每个分段最大 Token 上限: {max_chunk_tokens}")
            chunks = self._chunk_data_efficiently(cleaned_records, max_chunk_tokens)
            summaries_dir = run_output_dir / "summaries"
            summaries_dir.mkdir(exist_ok=True)
            log_with_timestamp(f"  - 分段摘要将保存至: {summaries_dir}")
            chunk_prompt_template = self.task_template["chunk_summary_prompt"]
            system_prompt = self.task_template.get("system_prompt")
            summaries = []
            api_delay = self.provider_config.get("api_call_delay_seconds", 0)
            for i, chunk in enumerate(chunks):
                summary = self.llm_connector.generate(
                    user_prompt=chunk_prompt_template.format(chunk_text=chunk),
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                )
                summaries.append(summary)
                summary_filename = f"{i+1:02d}_summary.txt"
                summary_path = summaries_dir / summary_filename
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                log_with_timestamp(f"  - ✅ 已保存摘要 {summary_filename}")
                if i < len(chunks) - 1 and api_delay > 0:
                    log_with_timestamp(
                        f"⏸️ 检测到 {self.llm_provider_name} 需要延迟调用，暂停 {api_delay} 秒..."
                    )
                    time.sleep(api_delay)
            llm_context = "\n\n".join(summaries)
        self._generate_final_report(
            llm_context, run_output_dir, attachment_data, attachment_type
        )

    def run_from_summaries(
        self,
        summary_dir_path: Path,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        log_with_timestamp(f"🔄 进入二次处理模式，从文件夹加载摘要: {summary_dir_path}")
        if not summary_dir_path.is_dir():
            log_with_timestamp(f"❌ 错误: 提供的路径不是一个有效的文件夹。")
            return
        summary_files = sorted(summary_dir_path.glob("*.txt"))
        if not summary_files:
            log_with_timestamp(
                f"❌ 错误: 在 {summary_dir_path} 中没有找到任何 .txt 摘要文件。"
            )
            return
        log_with_timestamp(f"  - 找到 {len(summary_files)} 个摘要文件，正在读取...")
        summaries = []
        for file_path in summary_files:
            with open(file_path, "r", encoding="utf-8") as f:
                summaries.append(f.read())
        llm_context = "\n\n".join(summaries)
        run_output_dir = (
            Path(self.config["output_path"])
            / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{self.cli_args.task}_{self.llm_provider_name}_finetuned"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_with_timestamp(f"📂 本次二次处理会话目录已创建: {run_output_dir}")
        self._generate_final_report(
            llm_context, run_output_dir, attachment_data, attachment_type
        )


def load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log_with_timestamp(f"错误: 配置文件 '{path}' 未找到。")
        exit(1)
    except json.JSONDecodeError:
        log_with_timestamp(f"错误: 配置文件 '{path}' 格式不正确。")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screenpipe OCR 智能日报生成助手",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="开始时间 (格式: YYYY-MM-DDTHH:MM:SS)，默认为24小时前",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="结束时间 (格式: YYYY-MM-DDTHH:MM:SS)，默认为当前时间",
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="配置文件的路径"
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="选择使用的大语言模型 (例如: gemini, deepseek)。会覆盖配置文件中的设置。",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="daily_report",
        help="选择要执行的任务 (对应 config.json 中的 prompt_templates key)。\n例如: daily_report, tutorial_generator",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="设置 LLM 的 Temperature。会覆盖配置文件中的设置。",
    )
    parser.add_argument(
        "--attachment", type=str, help="提供一个附件的路径 (可以是文本文件或图片)。"
    )
    parser.add_argument(
        "--use_summaries_from",
        type=str,
        help="提供一个包含摘要文件的文件夹路径，跳过数据提取和分段摘要，直接生成最终报告。",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    llm_provider = args.llm or config.get("llm_provider")
    if llm_provider == "gemini":
        gemini_config = config.get("llm_config", {}).get("gemini", {})
        proxy = gemini_config.get("proxy")
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
            log_with_timestamp(f"🔧 全局代理已为 Gemini 设置: {proxy}")
    attachment_data = None
    attachment_type = None
    if args.attachment:
        attachment_path = Path(args.attachment)
        if not attachment_path.exists():
            log_with_timestamp(f"❌ 错误: 附件文件未找到: {args.attachment}")
            exit(1)
        suffix = attachment_path.suffix.lower()
        if suffix in [".txt", ".md", ".json", ".xml", ".py", ".js"]:
            try:
                with open(attachment_path, "r", encoding="utf-8") as f:
                    attachment_data = f.read()
                attachment_type = "text"
                log_with_timestamp(f"📄 已成功加载文本附件: {args.attachment}")
            except Exception as e:
                log_with_timestamp(f"❌ 错误: 读取文本附件失败: {e}")
                exit(1)
        elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            try:
                attachment_data = Image.open(attachment_path)
                attachment_type = "image"
                log_with_timestamp(f"🖼️ 已成功加载图片附件: {args.attachment}")
            except Exception as e:
                log_with_timestamp(f"❌ 错误: 读取图片附件失败: {e}")
                exit(1)
        else:
            log_with_timestamp(
                f"⚠️ 警告: 不支持的附件文件类型 '{suffix}'。附件将被忽略。"
            )

    generator = DailyReportGenerator(config=config, cli_args=args)

    if args.use_summaries_from:
        summary_dir = Path(args.use_summaries_from)
        generator.run_from_summaries(summary_dir, attachment_data, attachment_type)
    else:
        try:
            if args.start_time:
                start_dt_local = datetime.fromisoformat(args.start_time)
            else:
                start_dt_local = datetime.now() - timedelta(days=1)
            if args.end_time:
                end_dt_local = datetime.fromisoformat(args.end_time)
            else:
                end_dt_local = datetime.now()
            start_dt_utc = start_dt_local.astimezone(timezone.utc)
            end_dt_utc = end_dt_local.astimezone(timezone.utc)
            log_with_timestamp(
                f"查询时间范围 (本地): {start_dt_local.isoformat(timespec='seconds')} -> {end_dt_local.isoformat(timespec='seconds')}"
            )
            log_with_timestamp(
                f"查询时间范围 (UTC): {start_dt_utc.isoformat(timespec='seconds')} -> {end_dt_utc.isoformat(timespec='seconds')}"
            )
        except ValueError:
            log_with_timestamp(
                "错误: 时间格式不正确。请使用 'YYYY-MM-DDTHH:MM:SS' 格式。"
            )
            exit(1)
        generator.run(
            start_time=start_dt_utc,
            end_time=end_dt_utc,
            attachment_data=attachment_data,
            attachment_type=attachment_type,
        )
