# main.py
import os
import json
import argparse
import difflib
import sqlite3
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# --- 动态导入检查 ---
try:
    import tiktoken
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai
    from openai import OpenAI
    import transformers
except ImportError as e:
    print(f"依赖库导入失败: {e}")
    print("请确保已通过 'pip install -r requirements.txt' 安装所有依赖。")
    exit(1)


# --- 模块 1.6: LLM 接口模块 (已升级) ---
class LLMConnector:
    def __init__(self, provider_name: str, provider_config: Dict[str, Any]):
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.client = self._initialize_client()
        print(
            f"🤖 LLM 连接器已为 provider 初始化: {self.provider_name} (模型: {self.provider_config.get('model_name')})"
        )

    def _initialize_client(self):
        if self.provider_name == "gemini":
            proxy = self.provider_config.get("proxy")
            if proxy:
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
                print(f"🔧 已为 Gemini 设置网络代理: {proxy}")

            genai.configure(api_key=self.provider_config["api_key"])
            return genai.GenerativeModel(self.provider_config["model_name"])

        elif self.provider_name == "deepseek":
            return OpenAI(
                api_key=self.provider_config["api_key"],
                base_url=self.provider_config["base_url"],
            )
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider_name}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        print(f"\n🚀 发起真实 LLM API 调用 (Temperature: {temperature})...")
        try:
            if self.provider_name == "gemini":
                # Gemini 通过 GenerationConfig 传递 temperature
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature
                )
                # 将 system_prompt (如果存在) 附加到 user_prompt 前面
                full_prompt = (
                    f"{system_prompt}\n\n---\n\n{user_prompt}"
                    if system_prompt
                    else user_prompt
                )
                response = self.client.generate_content(
                    full_prompt, generation_config=generation_config
                )
                return response.text

            elif self.provider_name == "deepseek":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.provider_config["model_name"],
                    temperature=temperature,
                )
                return chat_completion.choices[0].message.content

        except Exception as e:
            print(f"❌ LLM API 调用失败: {e}")
            return f"[LLM 调用错误: {e}]"
        return "[LLM 调用返回空]"


# --- 模块 1.1: 数据获取模块 ---
class DataFetcher:
    # ... 此部分代码无变化 ...
    def __init__(self):
        self.db_path = self._get_db_path()
        if not self.db_path or not self.db_path.exists():
            print(f"❌ 错误: 未能找到 Screenpipe 数据库。预期路径: {self.db_path}")
            exit(1)
        print(f"🔍 成功定位 Screenpipe 数据库: {self.db_path}")

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
        print(
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
            print(f"✅ 成功获取 {len(records)} 条 OCR 记录。")
            return records
        except sqlite3.Error as e:
            print(f"❌ 数据库查询失败: {e}")
            return []


# --- 核心应用类 (已升级) ---
class DailyReportGenerator:
    def __init__(self, config_path: str, cli_args: argparse.Namespace):
        self.config = self._load_config(config_path)
        self.data_fetcher = DataFetcher()
        self.cli_args = cli_args

        # 决定 LLM Provider
        if cli_args.llm and cli_args.llm in self.config["llm_config"]:
            self.llm_provider_name = cli_args.llm
            print(f"🔧 已通过命令行参数选择 LLM: {self.llm_provider_name}")
        else:
            self.llm_provider_name = self.config["llm_provider"]
            if cli_args.llm:
                print(
                    f"⚠️ 警告: 命令行指定的 LLM '{cli_args.llm}' 在 config.json 中未配置，将使用默认值: '{self.llm_provider_name}'"
                )
        self.provider_config = self.config["llm_config"][self.llm_provider_name]

        # 初始化 LLM 连接器
        self.llm_connector = LLMConnector(
            provider_name=self.llm_provider_name, provider_config=self.provider_config
        )

        # *** 关键变更: 加载指定的任务模板 ***
        self.task_template = self._load_task_template(cli_args.task)

        # *** 关键变更: 决定最终的 Temperature ***
        # 优先级: 命令行 > 配置文件
        if cli_args.temperature is not None:
            self.temperature = cli_args.temperature
            print(f"🔧 已通过命令行参数设置 Temperature: {self.temperature}")
        else:
            self.temperature = self.provider_config.get("temperature", 0.7)  # 默认 0.7

        # 加载 NLP 模型和 Tokenizer
        print("正在加载 NLP 模型...")
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = self._load_tokenizer()
        print("模型加载完成。")

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误: 配置文件 '{path}' 未找到。")
            exit(1)
        except json.JSONDecodeError:
            print(f"错误: 配置文件 '{path}' 格式不正确。")
            exit(1)

    def _load_task_template(self, task_name: str) -> Dict[str, str]:
        templates = self.config.get("prompt_templates", {})
        if task_name not in templates:
            print(f"❌ 错误: 任务 '{task_name}' 在配置文件中未定义。")
            print("可用的任务有:", list(templates.keys()))
            exit(1)
        print(
            f"🚀 已选择任务: '{task_name}' - {templates[task_name].get('description')}"
        )
        return templates[task_name]

    def _load_tokenizer(self):
        if self.llm_provider_name == "deepseek":
            tokenizer_path = self.provider_config.get("tokenizer_path")
            if tokenizer_path and os.path.isdir(tokenizer_path):
                try:
                    print(
                        f"  - 正在从本地路径加载 DeepSeek Tokenizer: {tokenizer_path}"
                    )
                    return transformers.AutoTokenizer.from_pretrained(
                        tokenizer_path, trust_remote_code=True
                    )
                except Exception as e:
                    print(
                        f"  - ❌ 错误: 加载 DeepSeek Tokenizer 失败: {e}。将回退到 tiktoken。"
                    )
        print("  - 正在加载 tiktoken 作为通用 Tokenizer。")
        return tiktoken.get_encoding("cl100k_base")

    def _clean_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ... 此部分代码无变化 ...
        if not records:
            return []
        print("🧹 开始数据清洗...")
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
        print(
            f"✅ 清洗完成。记录从 {len(records)} 条减少到 {len(cleaned_records)} 条。"
        )
        return cleaned_records

    def _estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def run(self, start_time: datetime, end_time: datetime):
        raw_records = self.data_fetcher.fetch_data(start_time, end_time)
        cleaned_records = self._clean_data(raw_records)
        if not cleaned_records:
            print("ℹ️ 清洗后无有效数据，无法生成报告。")
            return

        full_text = "\n\n---\n\n".join(
            [f"Timestamp: {r['captured_at']}\n{r['text']}" for r in cleaned_records]
        )
        total_tokens = self._estimate_tokens(full_text)
        print(f"📊 清洗后总 Token 数估算: {total_tokens}")

        llm_context = ""
        llm_context_window = self.provider_config["context_window"]
        effective_direct_summary_threshold = min(
            self.config["direct_summary_threshold"], llm_context_window
        )

        if total_tokens < effective_direct_summary_threshold:
            print("📈 Token 总数小于阈值，直接生成最终报告。")
            llm_context = full_text
        else:
            print("📉 Token 总数超过阈值，执行分段摘要流程。")
            available_space = llm_context_window - self.config["token_headroom"]
            if self.llm_provider_name == "deepseek" and isinstance(
                self.tokenizer, transformers.PreTrainedTokenizerFast
            ):
                max_chunk_tokens = available_space
                print(f"  - 使用精确的 DeepSeek Tokenizer，无需安全系数。")
            else:
                safety_margin = self.config.get("tokenizer_safety_margin", 1.0)
                max_chunk_tokens = int(available_space * safety_margin)
                print(f"  - 使用 tiktoken 估算，应用安全系数 ({safety_margin})...")
            print(f"  - 每个分段最大 Token 上限: {max_chunk_tokens}")

            chunks, current_chunk_text, current_chunk_tokens = [], "", 0
            for record in cleaned_records:
                record_text = f"Timestamp: {record['captured_at']}\n{record['text']}"
                record_tokens = self._estimate_tokens(record_text)
                if (
                    current_chunk_tokens + record_tokens > max_chunk_tokens
                    and current_chunk_text
                ):
                    chunks.append(current_chunk_text)
                    current_chunk_text, current_chunk_tokens = "", 0
                current_chunk_text += record_text + "\n\n---\n\n"
                current_chunk_tokens += record_tokens
            if current_chunk_text:
                chunks.append(current_chunk_text)

            print(f"  - 数据被分为 {len(chunks)} 个段落。")

            # *** 关键变更: 使用模板中的提示词 ***
            chunk_prompt_template = self.task_template["chunk_summary_prompt"]
            system_prompt = self.task_template.get("system_prompt")

            summaries = [
                self.llm_connector.generate(
                    user_prompt=chunk_prompt_template.format(chunk_text=chunk),
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                )
                for chunk in chunks
            ]
            llm_context = "\n\n".join(summaries)

        print("🖋️ 开始生成最终报告...")
        final_prompt_template = self.task_template["final_report_prompt"]
        final_report = self.llm_connector.generate(
            user_prompt=final_prompt_template.format(all_summaries=llm_context),
            system_prompt=system_prompt,
            temperature=self.temperature,
        )

        # *** 关键变更: 生成动态文件名 ***
        output_dir = self.config["output_path"]
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_filename = (
            f"{timestamp_str}_{self.cli_args.task}_{self.llm_provider_name}.md"
        )
        report_path = os.path.join(output_dir, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\n🎉 成功！报告已保存至: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screenpipe OCR 智能日报生成助手",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="开始时间 (格式: YYYY-MM-DDTHH:MM:SS)，默认为24小时前",
        default=(datetime.now() - timedelta(days=1)).isoformat(timespec="seconds"),
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="结束时间 (格式: YYYY-MM-DDTHH:MM:SS)，默认为当前时间",
        default=datetime.now().isoformat(timespec="seconds"),
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="配置文件的路径"
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="选择使用的大语言模型 (例如: gemini, deepseek)。会覆盖配置文件中的设置。",
    )

    # *** 新增命令行参数 ***
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

    args = parser.parse_args()

    try:
        start_dt, end_dt = datetime.fromisoformat(
            args.start_time
        ), datetime.fromisoformat(args.end_time)
    except ValueError:
        print("错误: 时间格式不正确。请使用 'YYYY-MM-DDTHH:MM:SS' 格式。")
        exit(1)

    generator = DailyReportGenerator(config_path=args.config, cli_args=args)
    generator.run(start_time=start_dt, end_time=end_dt)
