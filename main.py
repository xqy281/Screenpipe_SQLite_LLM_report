# main.py
import os
import json
import argparse
import difflib
import sqlite3
import platform
import time
import math
import threading
import random
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import concurrent.futures

# --- 动态导入检查 ---
try:
    import tiktoken
    from sentence_transformers import SentenceTransformer, util
    from google import genai
    from google.genai import types, errors as google_genai_errors
    from openai import OpenAI
    import transformers
    from PIL import Image
except ImportError as e:
    print(f"依赖库导入失败: {e}")
    print("请确保已通过 'pip install -r requirements.txt' 安装了所有依赖。")
    print("特别是，请确认您安装的是 'google-genai' 而不是 'google-generativeai'。")
    exit(1)


# --- 带时间戳的日志函数 ---
def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# [FINAL IMPLEMENTATION] 全局速率与并发控制中心
class RateControlManager:
    def __init__(self, rpm_limit: int, tpm_limit: int, concurrency_limit: int):
        self.rpm_limit = rpm_limit if rpm_limit > 0 else float('inf')
        self.tpm_limit = tpm_limit if tpm_limit > 0 else float('inf')
        
        self.request_timestamps = deque()
        self.token_usage = deque()
        self.cooldown_until = time.monotonic()
        
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(concurrency_limit)
        
        log_with_timestamp(f"✅ 全局速率控制中心已初始化: RPM={self.rpm_limit}, TPM={self.tpm_limit}, Concurrency={concurrency_limit}")

    def _prune(self):
        now = time.monotonic()
        one_minute_ago = now - 60
        while self.request_timestamps and self.request_timestamps[0] < one_minute_ago:
            self.request_timestamps.popleft()
        while self.token_usage and self.token_usage[0][0] < one_minute_ago:
            self.token_usage.popleft()

    def acquire(self, tokens_to_send: int):
        self.semaphore.acquire()
        try:
            while True:
                with self.lock:
                    now = time.monotonic()
                    
                    if now < self.cooldown_until:
                        sleep_time = self.cooldown_until - now
                    else:
                        sleep_time = 0

                    if sleep_time > 0:
                        pass
                    else:
                        self._prune()
                        current_rpm = len(self.request_timestamps)
                        current_tpm = sum(count for _, count in self.token_usage)
                        
                        if current_rpm < self.rpm_limit and (current_tpm + tokens_to_send) <= self.tpm_limit:
                            return
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.1) 
        except Exception:
            self.semaphore.release()
            raise

    def release(self, tokens_sent: int):
        with self.lock:
            now = time.monotonic()
            self.request_timestamps.append(now)
            self.token_usage.append((now, tokens_sent))
        self.semaphore.release()

    def trigger_cooldown(self, delay: float):
        with self.lock:
            cooldown_end_time = time.monotonic() + delay
            self.cooldown_until = max(self.cooldown_until, cooldown_end_time)
            log_with_timestamp(f"  - 🚨 [RateCtrl] 收到429！启动全局冷却，所有新请求将暂停 {delay:.1f} 秒。")


# --- 模块 1.6: LLM 接口模块 ---
class BaseLLMConnector:
    def __init__(
        self,
        provider_config: Dict[str, Any],
        debug_mode: bool = False,
        run_output_dir: Optional[Path] = None,
        rate_control_manager: Optional[RateControlManager] = None,
        dry_run: bool = False,
    ):
        self.provider_config = provider_config
        self.model_name = provider_config.get("model_name")
        self.debug_mode = debug_mode
        self.run_output_dir = run_output_dir
        self.rate_control_manager = rate_control_manager
        self.dry_run = dry_run
        
        if self.dry_run:
            log_with_timestamp("🚱 Dry-run 模式已激活。将跳过付费的LLM API调用。")

        if (self.debug_mode or self.dry_run) and self.run_output_dir:
            self.snapshots_dir = self.run_output_dir / "prompt_snapshots"
            self.snapshots_dir.mkdir(exist_ok=True)
            log_with_timestamp(
                f"🔍 Prompt快照功能已激活，将保存至: {self.snapshots_dir}"
            )
        else:
            self.snapshots_dir = None

        self.client = self._initialize_client()
        log_with_timestamp(
            f"🤖 {self.__class__.__name__} 已初始化 (模型: {self.model_name})"
        )

    def _save_prompt_snapshot(
        self, prompt_name: str, system_prompt: Optional[str], user_prompt: Any
    ):
        if not self.snapshots_dir:
            return
        
        user_prompt_str = ""
        if isinstance(user_prompt, list):
            for item in user_prompt:
                if isinstance(item, str):
                    user_prompt_str += item
        else:
            user_prompt_str = str(user_prompt)

        snapshot_content = f"--- SYSTEM PROMPT ---\n\n{system_prompt}\n\n"
        snapshot_content += f"--- USER PROMPT ---\n\n{user_prompt_str}"
        snapshot_path = self.snapshots_dir / f"{prompt_name}.txt"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(snapshot_content)

    def _initialize_client(self):
        raise NotImplementedError

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
        prompt_name: str = "prompt",
    ) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError


class GeminiConnector(BaseLLMConnector):
    def _initialize_client(self):
        # 即使在 dry-run 模式下也需要客户端来调用免费的 count_tokens
        api_key = self.provider_config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key not found in config.")
        
        os.environ['GOOGLE_API_KEY'] = api_key
        log_with_timestamp("  - 🔑 已通过环境变量设置 Gemini API Key。")
        
        try:
            client = genai.Client()
            return client
        except Exception as e:
            print(f"❌ 初始化 Gemini 客户端时发生未知错误: {e}")
            print("   请确认您的 'google-genai' 库已正确安装且版本兼容。")
            exit(1)

    def _parse_retry_delay(self, error: Exception) -> Optional[float]:
        if not isinstance(error, google_genai_errors.APIError):
            return None
        
        error_str = str(error)
        try:
            retry_info_str = "'@type': 'type.googleapis.com/google.rpc.RetryInfo'"
            if retry_info_str in error_str:
                delay_str_start = error_str.find("'retryDelay': '", error_str.find(retry_info_str))
                if delay_str_start != -1:
                    delay_str_start += len("'retryDelay': '")
                    delay_str_end = error_str.find("s'", delay_str_start)
                    delay_seconds = float(error_str[delay_str_start:delay_str_end])
                    return delay_seconds
        except (ValueError, TypeError):
            return None
        return None

    def _generate_with_manual_retry(self, **kwargs) -> types.GenerateContentResponse:
        max_retries = 7
        initial_delay = 2.0
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(**kwargs)
                if attempt > 0:
                    log_with_timestamp(f"  - ✅ 重试成功 (在第 {attempt + 1} 次尝试)!")
                return response
            
            except google_genai_errors.APIError as e:
                last_exception = e
                should_retry = False
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)

                error_code = getattr(e, 'code', None)

                if error_code == 429:
                    server_suggested_delay = self._parse_retry_delay(e)
                    if server_suggested_delay is not None:
                        delay = server_suggested_delay + random.uniform(0, 1)
                    
                    log_with_timestamp(f"  - ⚠️ API速率限制 (429), 尝试 {attempt + 1}/{max_retries}。")
                    if self.rate_control_manager and server_suggested_delay is not None:
                        self.rate_control_manager.trigger_cooldown(delay)
                    else:
                        log_with_timestamp(f"     将使用局部退避，在 {delay:.1f} 秒后重试...")
                    should_retry = True

                elif error_code in [500, 502, 503]:
                     log_with_timestamp(f"  - ⚠️ API服务器错误 (Code: {error_code}, 尝试 {attempt + 1}/{max_retries})。将使用局部退避，在 {delay:.1f} 秒后重试...")
                     should_retry = True
                
                if should_retry and attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    break

            except Exception as e:
                last_exception = e
                if ("SSL" in str(e) or "EOF" in str(e)) and attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                    log_with_timestamp(f"  - ⚠️ 捕获到底层网络错误, 尝试 {attempt + 1}/{max_retries}。将在 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    log_with_timestamp(f"❌ API调用期间发生意外的本地编程错误: {e}")
                    raise e

        log_with_timestamp(f"  - ❌ 达到最大重试次数。最后一次错误: {last_exception}")
        raise last_exception


    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
        prompt_name: str = "prompt",
    ) -> str:
        contents = [user_prompt]
        if attachment_data:
            if attachment_type == "image":
                contents.append(attachment_data)
            elif attachment_type == "text":
                contents[0] = (
                    f"### 补充上下文:\n{attachment_data}\n\n### 主要任务:\n{user_prompt}"
                )

        self._save_prompt_snapshot(prompt_name, system_prompt, contents)

        if self.dry_run:
            return ""

        config = types.GenerateContentConfig(
            temperature=temperature,
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ],
            system_instruction=system_prompt
        )

        try:
            response = self._generate_with_manual_retry(
                model=f"models/{self.model_name}",
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            return f"[LLM_ERROR]: {e}"

    def count_tokens(self, text: str) -> int:
        # count_tokens 是免费的，所以即使在 dry-run 模式下也执行以获得精确分块
        if not self.client:
             log_with_timestamp("  - ⚠️ Dry-run 模式下无客户端，使用粗算 Token。")
             return len(tiktoken.get_encoding("cl100k_base").encode(text))
            
        try:
            response = self.client.models.count_tokens(
                model=f"models/{self.model_name}",
                contents=[text]
            )
            return response.total_tokens
        except Exception as e:
            log_with_timestamp(f"❌ Gemini Token 计数失败: {e}. 将使用粗算方法。")
            return len(text) // 4


class DeepSeekConnector(BaseLLMConnector):
    def _initialize_client(self):
        if self.dry_run:
            return None
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
        prompt_name: str = "prompt",
    ) -> str:
        if attachment_type == "text" and attachment_data:
            user_prompt = (
                f"### 补充上下文:\n{attachment_data}\n\n### 主要任务:\n{user_prompt}"
            )

        self._save_prompt_snapshot(prompt_name, system_prompt, user_prompt)

        if self.dry_run:
            return ""

        log_with_timestamp(f"🚀 发起 DeepSeek API 调用 (Temperature: {temperature})...")
        if attachment_type == "image":
            log_with_timestamp(
                "❌ 错误: DeepSeek 的 OpenAI 兼容 API 当前不支持直接的图像输入。"
            )
            return "[错误: 此模型 API 不支持图像输入]"

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
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
        raise NotImplementedError("DeepSeek token counting should be handled locally.")


class LLMConnectorFactory:
    @staticmethod
    def create(
        provider_name: str,
        provider_config: Dict[str, Any],
        debug_mode: bool,
        run_output_dir: Path,
        rate_control_manager: Optional[RateControlManager] = None,
        dry_run: bool = False,
    ) -> BaseLLMConnector:
        if provider_name == "gemini":
            return GeminiConnector(provider_config, debug_mode, run_output_dir, rate_control_manager, dry_run)
        elif provider_name == "deepseek":
            return DeepSeekConnector(provider_config, debug_mode, run_output_dir, dry_run=dry_run)
        else:
            raise ValueError(f"不支持的 LLM provider: {provider_name}")


# --- 模块 1.1: 数据获取模块 ---
class DataFetcher:
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
        query = "SELECT frm.id as frame_id, ocr.text, frm.timestamp AS captured_at FROM frames AS frm JOIN ocr_text AS ocr ON frm.id = ocr.frame_id WHERE frm.timestamp >= ? AND frm.timestamp <= ? ORDER BY frm.timestamp ASC;"
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
        self.llm_connector = None
        self.task_template = self._load_task_template(cli_args.task)
        if cli_args.temperature is not None:
            self.temperature = cli_args.temperature
            log_with_timestamp(
                f"🔧 已通过命令行参数设置 Temperature: {self.temperature}"
            )
        else:
            self.temperature = self.provider_config.get("temperature", 0.7)
        
        self.rate_control_manager = None
        if self.llm_provider_name == 'gemini' and not self.cli_args.dry_run:
            rate_limit_config = self.provider_config.get("rate_limiting", {})
            self.rate_control_manager = RateControlManager(
                rpm_limit=rate_limit_config.get("rpm", 15),
                tpm_limit=rate_limit_config.get("tpm", 100000),
                concurrency_limit=rate_limit_config.get("concurrency", 15)
            )

        log_with_timestamp("正在加载 NLP 模型...")
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_tokenizers()
        log_with_timestamp("模型加载完成。")

    def _initialize_llm_connector(self, run_output_dir: Path):
        if self.llm_connector is None:
            self.llm_connector = LLMConnectorFactory.create(
                provider_name=self.llm_provider_name,
                provider_config=self.provider_config,
                debug_mode=self.cli_args.debug,
                run_output_dir=run_output_dir,
                rate_control_manager=self.rate_control_manager,
                dry_run=self.cli_args.dry_run,
            )

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
            else:
                log_with_timestamp(f"  - ⚠️ 未找到 DeepSeek Tokenizer，精算将不可用。")
        log_with_timestamp(
            f"  - 精算方式已确定: {'local_exact' if self.precise_tokenizer else 'api'}"
        )

    def _clean_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        log_with_timestamp("🧹 开始数据清洗 (sentence-transformers + difflib)...")
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
        # --- 最终修复：移除此处的 dry-run 判断，完全委托给 connector ---
        if self.llm_provider_name == "gemini":
            if not self.llm_connector:
                raise RuntimeError(
                    "LLMConnector not initialized before counting tokens."
                )
            return self.llm_connector.count_tokens(text)
        elif self.llm_provider_name == "deepseek" and self.precise_tokenizer:
            return len(self.precise_tokenizer.encode(text))
        return self._estimate_rough_tokens(text)

    def _estimate_rough_tokens(self, text: str) -> int:
        return len(self.rough_tokenizer.encode(text))

    def _chunk_data_efficiently(
        self, records: List[Dict[str, Any]], max_chunk_tokens: int
    ) -> List[str]:
        log_with_timestamp("⏳ 正在高效地对数据进行分段 (API精算模式)...")
        
        log_with_timestamp("  - 步骤1: 正在对所有记录进行Token粗算预计算...")
        utc_plus_8 = timezone(timedelta(hours=8))
        records_with_metadata = []
        for record in records:
            try:
                utc_time = datetime.fromisoformat(
                    record["captured_at"].replace("Z", "+00:00")
                )
                local_time_str = utc_time.astimezone(utc_plus_8).strftime(
                    "%Y-%m-%d %H:%M:%S (UTC+8)"
                )
            except (ValueError, TypeError):
                local_time_str = record["captured_at"]
            record_text_for_llm = f"Timestamp: {local_time_str}\n{record['text']}"
            rough_token_count = self._estimate_rough_tokens(record_text_for_llm)
            records_with_metadata.append(
                {
                    "text_for_llm": record_text_for_llm,
                    "rough_tokens": rough_token_count,
                }
            )
        log_with_timestamp("  - ✅ 预计算完成。")

        chunks = []
        start_index = 0
        while start_index < len(records_with_metadata):
            current_rough_tokens = 0
            end_index = start_index
            while end_index < len(records_with_metadata):
                current_rough_tokens += records_with_metadata[end_index]["rough_tokens"]
                if current_rough_tokens > max_chunk_tokens and end_index > start_index:
                    current_rough_tokens -= records_with_metadata[end_index][
                        "rough_tokens"
                    ]
                    break
                end_index += 1
            
            current_chunk_llm_texts = [
                rec["text_for_llm"]
                for rec in records_with_metadata[start_index:end_index]
            ]
            temp_chunk_text = "\n\n---\n\n".join(current_chunk_llm_texts)
            
            log_with_timestamp(
                f"  - 粗算打包完成一个块 (记录 {start_index}-{end_index-1}，粗算 {current_rough_tokens})，开始API精算..."
            )
            precise_tokens = self._estimate_precise_tokens(temp_chunk_text)
            log_with_timestamp(f"  - API精算结果: {precise_tokens} tokens。")

            while (
                precise_tokens > max_chunk_tokens and len(current_chunk_llm_texts) > 1
            ):
                log_with_timestamp(
                    f"  - 精算后仍超限 (溢出 {precise_tokens - max_chunk_tokens} tokens)，开始动态移除..."
                )
                token_overflow = precise_tokens - max_chunk_tokens
                avg_tokens_per_record = precise_tokens / len(current_chunk_llm_texts)
                num_to_remove = (
                    math.ceil(token_overflow / avg_tokens_per_record)
                    if avg_tokens_per_record > 0
                    else 1
                )
                num_to_remove = max(1, num_to_remove)
                log_with_timestamp(
                    f"  - 动态计算：预估需移除 {num_to_remove} 条记录..."
                )
                current_chunk_llm_texts = current_chunk_llm_texts[:-num_to_remove]
                end_index -= num_to_remove
                temp_chunk_text = "\n\n---\n\n".join(current_chunk_llm_texts)
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
            prompt_name="final_report_prompt",
        )
        report_path = run_output_dir / "final_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        log_with_timestamp(f"\n🎉 成功！报告已保存至: {report_path}")

    def _process_records(
        self,
        cleaned_records: List[Dict[str, Any]],
        run_output_dir: Path,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        if not cleaned_records:
            log_with_timestamp("ℹ️ 无有效数据可处理，无法生成报告。")
            return
        self._initialize_llm_connector(run_output_dir)
        log_with_timestamp("📉 执行标准分段摘要流程...")
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
        
        log_with_timestamp(f"  - 开始对 {len(chunks)} 个数据块进行摘要（采用智能重试，无固定延迟）...")
        for i, chunk in enumerate(chunks):
            log_with_timestamp(f"  - 正在处理摘要 {i+1}/{len(chunks)}...")
            prompt_name = f"{i+1:02d}_chunk_summary_prompt"
            summary = self.llm_connector.generate(
                user_prompt=chunk_prompt_template.format(chunk_text=chunk),
                system_prompt=system_prompt,
                temperature=self.temperature,
                prompt_name=prompt_name,
            )
            summaries.append(summary)
            summary_filename = f"{i+1:02d}_summary.txt"
            summary_path = summaries_dir / summary_filename
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            log_with_timestamp(f"  - ✅ 已保存摘要 {summary_filename}")

        llm_context = "\n\n".join(summaries)
        self._generate_final_report(
            llm_context, run_output_dir, attachment_data, attachment_type
        )

    # --- 实验性功能：三步智能提纯法 ---
    def _run_llm_refinement_flow(
        self,
        records: List[Dict[str, Any]],
        run_output_dir: Path,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        log_with_timestamp("🧪 开始执行实验性LLM智能提纯流程...")
        self._initialize_llm_connector(run_output_dir)

        # 步骤一：智能分类
        classified_records = self._classify_records_with_llm(records, run_output_dir)
        if not classified_records:
            log_with_timestamp("  - 步骤一（分类）未产生有效数据，流程中止。")
            return

        # 步骤二：编程分组
        record_groups = self._group_classified_records(classified_records)
        if not record_groups:
            log_with_timestamp("  - 步骤二（分组）未产生有效数据，流程中止。")
            return

        # 步骤三：上下文感知总结
        final_context = self._summarize_record_groups(record_groups, run_output_dir)
        if not final_context:
            log_with_timestamp("  - 步骤三（总结）未产生有效内容，流程中止。")
            return

        # 使用提纯后的上下文生成最终报告
        self._generate_final_report(final_context, run_output_dir, attachment_data, attachment_type)

    def _classify_single_record(self, record: Dict[str, Any]) -> (Optional[str], Optional[str], int):
        """Classifies a single record and returns (frame_id, category, total_tokens_used)."""
        classification_prompt_template = self.config.get("llm_refinement_config", {}).get("classification_prompt")
        if not classification_prompt_template:
            raise ValueError("在 config.json 中未找到 'llm_refinement_config.classification_prompt'")

        frame_id = str(record.get("frame_id"))
        
        json_data_for_single_record = json.dumps([
            {"frame_id": frame_id, "text": record["text"]}
        ], ensure_ascii=False)

        prompt = classification_prompt_template.format(json_data=json_data_for_single_record)
        
        response = self.llm_connector._generate_with_manual_retry(
            model=f"models/{self.llm_connector.model_name}",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        response_str = response.text
        total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0

        try:
            response_str = response_str.strip().replace("```json", "").replace("```", "").strip()
            classification_result = json.loads(response_str)
            category = classification_result.get(frame_id, "Noise/UI")
            
            valid_categories = ['EmbeddedCoding', 'BuildAndCompile', 'DeploymentAndDebugging', 'FirmwareValidation', 'VersionControl', 'APIDebugging', 'HardwareAndRF', 'ResearchAndAI', 'Noise/UI']
            if category in valid_categories:
                return frame_id, category, total_tokens
            else:
                log_with_timestamp(f"  - ⚠️ 记录 {frame_id} 返回了无效分类 '{category}'，将标记为Noise/UI。")
                return frame_id, "Noise/UI", total_tokens
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log_with_timestamp(f"  - ❌ 解析记录 {frame_id} 的分类结果失败: {e}")
            log_with_timestamp(f"  - 返回的原始文本: {response_str[:200]}...")
            return frame_id, "Noise/UI", total_tokens

    def _classify_records_with_llm(self, records: List[Dict[str, Any]], run_output_dir: Path) -> List[Dict[str, Any]]:
        log_with_timestamp("  - 步骤一：原子化并发分类 (带全局速率控制)...")
        
        for i, record in enumerate(records):
            if 'frame_id' not in record or not record['frame_id']:
                record['frame_id'] = f"generated_id_{i}"

        all_classifications = {}
        
        max_workers = 1 if self.cli_args.dry_run else (self.rate_control_manager.semaphore._value * 2 if self.rate_control_manager else 50)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            def task_wrapper(record):
                estimated_input_tokens = 0
                actual_tokens_used = 0
                if self.rate_control_manager:
                    estimated_input_tokens = self._estimate_rough_tokens(record['text'])
                    self.rate_control_manager.acquire(estimated_input_tokens)
                
                try:
                    frame_id, category, actual_tokens_used = self._classify_single_record(record)
                    return frame_id, category
                finally:
                    if self.rate_control_manager:
                        release_tokens = actual_tokens_used if actual_tokens_used > 0 else estimated_input_tokens
                        self.rate_control_manager.release(release_tokens)

            future_to_record = {
                executor.submit(task_wrapper, record): record
                for record in records
            }
            
            processed_count = 0
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    frame_id, category = future.result()
                    if frame_id and category:
                        all_classifications[frame_id] = category
                except Exception as exc:
                    log_with_timestamp(f"  - ❌ 记录 {record.get('frame_id')} 分类任务失败: {exc}")
                
                processed_count += 1
                if processed_count % 100 == 0 or processed_count == len(records):
                    log_with_timestamp(f"  - ...已处理 {processed_count}/{len(records)} 条记录...")

        log_with_timestamp(f"  - ✅ 所有分类任务处理完毕，共获得 {len(all_classifications)} 条分类映射。")
        
        final_records = []
        for record in records:
            frame_id = str(record.get("frame_id"))
            category = all_classifications.get(frame_id, "Noise/UI")
            record["activity_type"] = category
            final_records.append(record)
        
        classified_path = run_output_dir / "classified_records.json"
        with open(classified_path, "w", encoding="utf-8") as f:
            json.dump(final_records, f, ensure_ascii=False, indent=2)
        log_with_timestamp(f"  - 💾 分类后数据已保存至: {classified_path}")
        
        return final_records

    def _group_classified_records(self, classified_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        log_with_timestamp("  - 步骤二：编程分组...")
        if not classified_records:
            return []

        groups = []
        current_group = {
            "type": classified_records[0].get("activity_type", "Noise/UI"),
            "records": [classified_records[0]]
        }

        for record in classified_records[1:]:
            activity_type = record.get("activity_type", "Noise/UI")
            if activity_type == current_group["type"]:
                current_group["records"].append(record)
            else:
                if current_group["type"] != "Noise/UI":
                    groups.append(current_group)
                current_group = {"type": activity_type, "records": [record]}
        
        if current_group["type"] != "Noise/UI":
            groups.append(current_group)

        log_with_timestamp(f"  - ✅ 分组完成，数据被分为 {len(groups)} 个有效活动块。")
        return groups

    def _summarize_record_groups(self, groups: List[Dict[str, Any]], run_output_dir: Path) -> str:
        log_with_timestamp("  - 步骤三：并发上下文感知总结...")
        summarization_prompts = self.config.get("llm_refinement_config", {}).get("summarization_prompts", {})
        if not summarization_prompts:
            log_with_timestamp("  - ❌ 错误: 在config.json中未找到'llm_refinement_config.summarization_prompts'。")
            return ""

        summaries_dir = run_output_dir / "llm_refinement_summaries"
        summaries_dir.mkdir(exist_ok=True)

        tasks = []
        for i, group in enumerate(groups):
            if group.get("type", "Noise/UI") != "Noise/UI":
                tasks.append((group, i + 1))

        final_summaries = [None] * len(tasks)
        
        max_workers = 1 if self.cli_args.dry_run else (self.rate_control_manager.semaphore._value if self.rate_control_manager else 20)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            def task_wrapper(group, group_index):
                estimated_input_tokens = 0
                actual_tokens_used = 0
                context_text = "\n--- NEW_FRAME ---\n".join([r["text"] for r in group["records"]])

                if self.rate_control_manager:
                    estimated_input_tokens = self._estimate_rough_tokens(context_text)
                    self.rate_control_manager.acquire(estimated_input_tokens)
                
                try:
                    summary, actual_tokens_used = self._summarize_single_group(group, group_index, context_text)
                    return summary
                finally:
                    if self.rate_control_manager:
                        release_tokens = actual_tokens_used if actual_tokens_used > 0 else estimated_input_tokens
                        self.rate_control_manager.release(release_tokens)

            future_to_index = {
                executor.submit(task_wrapper, task_data[0], task_data[1]): i
                for i, task_data in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    summary_text = future.result()
                    final_summaries[index] = summary_text
                except Exception as exc:
                    log_with_timestamp(f"  - ❌ 总结任务 {index + 1} 失败: {exc}")
        
        valid_summaries = [s for s in final_summaries if s is not None]
        for i, summary_text in enumerate(valid_summaries):
             summary_path = summaries_dir / f"{i+1:02d}_summary.txt"
             with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

        log_with_timestamp(f"  - ✅ 总结完成，生成了 {len(valid_summaries)} 条有效活动摘要。")
        return "\n\n".join(valid_summaries)

    def _summarize_single_group(self, group: Dict[str, Any], group_index: int, context_text: str) -> (str, int):
        group_type = group.get("type", "default")
        summarization_prompts = self.config["llm_refinement_config"]["summarization_prompts"]
        prompt_template = summarization_prompts.get(group_type, summarization_prompts.get("default"))
        
        if not prompt_template:
            raise ValueError(f"No summary prompt found for type '{group_type}'")

        prompt = prompt_template.format(data=context_text)

        system_prompt = f"你是一个专家级的活动总结助理，当前正在分析一个 '{group_type}' 类型的活动。"
        
        config = types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=system_prompt
        )

        response = self.llm_connector._generate_with_manual_retry(
            model=f"models/{self.llm_connector.model_name}",
            contents=[prompt],
            config=config
        )
        
        summary = response.text
        total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0

        first_record = group['records'][0]
        utc_time = datetime.fromisoformat(first_record["captured_at"].replace("Z", "+00:00"))
        local_time_str = utc_time.astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S")

        formatted_summary = f"**活动: {group_type}** (从 {local_time_str} 开始)\n- {summary.strip()}"
        return formatted_summary, total_tokens

    def run(
        self,
        start_time_utc: datetime,
        end_time_utc: datetime,
        start_time_local: datetime,
        end_time_local: datetime,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        time_range_str = f"{start_time_local.strftime('%Y%m%d_%H%M')}-{end_time_local.strftime('%Y%m%d_%H%M')}"
        current_timestamp_str = datetime.now().strftime("%y-%m-%d %H_%M_%S")
        session_name = f"{time_range_str}_{self.cli_args.task}_{self.llm_provider_name}_{current_timestamp_str}"
        run_output_dir = Path(self.config["output_path"]) / session_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_with_timestamp(f"📂 本次运行会话目录已创建: {run_output_dir}")
        raw_records = self.data_fetcher.fetch_data(start_time_utc, end_time_utc)
        cleaned_records = self._clean_data(raw_records)
        cleaned_data_path = run_output_dir / "cleaned_data.json"
        with open(cleaned_data_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
        log_with_timestamp(f"💾 已清洗的数据已缓存至: {cleaned_data_path}")

        if self.cli_args.enable_llm_refinement:
            if self.llm_provider_name == 'gemini':
                self._run_llm_refinement_flow(
                    cleaned_records, run_output_dir, attachment_data, attachment_type
                )
            else:
                log_with_timestamp("⚠️ 警告: LLM智能提纯功能当前仅支持Gemini。将执行标准流程。")
                self._process_records(
                    cleaned_records, run_output_dir, attachment_data, attachment_type
                )
        else:
            self._process_records(
                cleaned_records, run_output_dir, attachment_data, attachment_type
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
        source_dir_name = summary_dir_path.name
        current_timestamp_str = datetime.now().strftime("%y-%m-%d %H_%M_%S")
        session_name = f"from_summaries_{source_dir_name}_{self.cli_args.task}_{self.llm_provider_name}_{current_timestamp_str}"
        run_output_dir = Path(self.config["output_path"]) / session_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_with_timestamp(f"📂 本次二次处理会话目录已创建: {run_output_dir}")
        self._initialize_llm_connector(run_output_dir)
        self._generate_final_report(
            llm_context, run_output_dir, attachment_data, attachment_type
        )

    def run_from_cleaned_data(
        self,
        cleaned_data_path: Path,
        attachment_data: Optional[Any] = None,
        attachment_type: Optional[str] = None,
    ):
        log_with_timestamp(
            f"🔄 进入缓存加载模式，从文件加载已清洗数据: {cleaned_data_path}"
        )
        try:
            with open(cleaned_data_path, "r", encoding="utf-8") as f:
                cleaned_records = json.load(f)
            log_with_timestamp(f"  - ✅ 成功加载 {len(cleaned_records)} 条已清洗记录。")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log_with_timestamp(f"❌ 错误: 加载缓存文件失败: {e}")
            return
        source_file_name = cleaned_data_path.stem
        current_timestamp_str = datetime.now().strftime("%y-%m-%d %H_%M_%S")
        session_name = f"from_cleaned_{source_file_name}_{self.cli_args.task}_{self.llm_provider_name}_{current_timestamp_str}"
        run_output_dir = Path(self.config["output_path"]) / session_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        log_with_timestamp(f"📂 本次缓存处理会话目录已创建: {run_output_dir}")
        
        if self.cli_args.enable_llm_refinement:
            if self.llm_provider_name == 'gemini':
                self._run_llm_refinement_flow(
                    cleaned_records, run_output_dir, attachment_data, attachment_type
                )
            else:
                log_with_timestamp("⚠️ 警告: LLM智能提纯功能当前仅支持Gemini。将执行标准流程。")
                self._process_records(
                    cleaned_records, run_output_dir, attachment_data, attachment_type
                )
        else:
            self._process_records(
                cleaned_records, run_output_dir, attachment_data, attachment_type
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
    parser.add_argument(
        "--load_cleaned_data",
        type=str,
        help="提供一个 cleaned_data.json 文件路径，跳过数据提取和清洗，直接进行分段和报告生成。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，将保存发送给LLM的完整Prompt快照。",
    )
    parser.add_argument(
        "--enable-llm-refinement",
        action="store_true",
        help="[实验性功能, Gemini专用] 启用三步LLM智能提纯流程，以获得更高质量的摘要。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成Prompt快照和空的摘要文件，不执行任何LLM API调用。",
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
    elif args.load_cleaned_data:
        cleaned_data_file = Path(args.load_cleaned_data)
        generator.run_from_cleaned_data(
            cleaned_data_file, attachment_data, attachment_type
        )
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
            start_time_utc=start_dt_utc,
            end_time_utc=end_dt_utc,
            start_time_local=start_dt_local,
            end_time_local=end_dt_local,
            attachment_data=attachment_data,
            attachment_type=attachment_type,
        )