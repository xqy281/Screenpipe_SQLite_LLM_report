# db_inspector.py
import sqlite3
import platform
import argparse
from pathlib import Path
from datetime import datetime, timezone


def log_message(message: str):
    """简单的日志打印函数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_db_path() -> Path | None:
    """根据操作系统类型获取数据库路径"""
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


def inspect_time_range(start_time_str: str, end_time_str: str):
    """
    连接到数据库，将本地时间输入转换为UTC，并查询指定时间范围的数据。
    """
    db_path = get_db_path()
    if not db_path or not db_path.exists():
        log_message(f"❌ 错误: 未能找到 Screenpipe 数据库。预期路径: {db_path}")
        return

    log_message(f"🔍 成功定位 Screenpipe 数据库: {db_path}")
    print("-" * 50)

    # 1. 将输入的本地时间字符串转换为UTC时间对象
    try:
        start_dt_local = datetime.fromisoformat(start_time_str)
        end_dt_local = datetime.fromisoformat(end_time_str)

        start_dt_utc = start_dt_local.astimezone(timezone.utc)
        end_dt_utc = end_dt_local.astimezone(timezone.utc)

        log_message("🕒 时间转换详情:")
        print(f"  - 本地开始时间 (输入): {start_dt_local.isoformat()}")
        print(f"  - 本地结束时间 (输入): {end_dt_local.isoformat()}")
        print(f"  - 转换为 UTC 开始时间: {start_dt_utc.isoformat()}")
        print(f"  - 转换为 UTC 结束时间: {end_dt_utc.isoformat()}")
        print("-" * 50)

    except ValueError:
        log_message("❌ 错误: 时间格式不正确。请使用 'YYYY-MM-DDTHH:MM:SS' 格式。")
        return

    # 2. 准备并显示SQL查询
    query = """
    SELECT
        ocr.text,
        frm.timestamp AS captured_at
    FROM
        frames AS frm
    JOIN
        ocr_text AS ocr ON frm.id = ocr.frame_id
    WHERE
        frm.timestamp >= ? AND frm.timestamp <= ?
    ORDER BY
        frm.timestamp ASC;
    """
    log_message("📄 将要执行的 SQL 查询:")
    print(query)
    print("-" * 50)

    # 3. 连接数据库并执行查询
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(query, (start_dt_utc.isoformat(), end_dt_utc.isoformat()))
            results = cursor.fetchall()

            log_message(
                f"📊 查询完成。在指定的UTC时间范围内共找到 {len(results)} 条记录。"
            )
            print("-" * 50)

            if not results:
                log_message(
                    "ℹ️ 提示: 如果记录数为0，请检查输入的时间范围是否确实有屏幕活动。"
                )
            else:
                log_message("📝 查询结果预览 (最多显示前10条):")
                for i, row in enumerate(results[:10]):
                    print(f"\n--- 记录 {i+1} ---")
                    print(f"  - captured_at (UTC): {row['captured_at']}")
                    # 截断长文本，方便预览
                    print(f"  - text (预览): {row['text'][:150]}...")
                if len(results) > 10:
                    print("\n...")
                    print(f"（还有 {len(results) - 10} 条记录未显示）")

    except sqlite3.Error as e:
        log_message(f"❌ 数据库查询失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Screenpipe 数据库时间范围查询验证工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--start_time",
        type=str,
        required=True,
        help="要查询的开始时间 (本地时间, 格式: YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        required=True,
        help="要查询的结束时间 (本地时间, 格式: YYYY-MM-DDTHH:MM:SS)",
    )
    args = parser.parse_args()

    inspect_time_range(args.start_time, args.end_time)
