# db_inspector.py
import sqlite3
import platform
from pathlib import Path


def get_db_path() -> Path | None:
    """根据操作系统类型获取数据库路径"""
    system = platform.system()
    home = Path.home()
    # 根据你的输出，Windows 路径是 .screenpipe 文件夹
    if system == "Windows":
        return home / ".screenpipe/db.sqlite"
    if system == "Darwin":  # macOS
        return home / "Library/Application Support/Screenpipe/db.sqlite"
    if system == "Linux":
        return home / ".config/Screenpipe/db.sqlite"
    return None


def inspect_database():
    """连接到数据库并打印出所有表名及其列信息"""
    db_path = get_db_path()
    if not db_path or not db_path.exists():
        print(f"❌ 错误: 未能找到 Screenpipe 数据库。预期路径: {db_path}")
        return

    print(f"🔍 正在检查数据库: {db_path}\n")

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()

            # 1. 列出所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                print("数据库中没有找到任何表。")
                return

            print("=" * 20)
            print("数据库中的所有表:")
            table_names = [table[0] for table in tables]
            print(table_names)
            print("=" * 20 + "\n")

            # 2. 检查每个表的结构
            for table_name in table_names:
                print(f"--- 表 '{table_name}' 的结构 ---")
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                if not columns:
                    print("  (无法获取列信息)")
                else:
                    for column in columns:
                        # (id, name, type, notnull, default_value, pk)
                        print(f"  - 列名: {column[1]}, 类型: {column[2]}")
                print("-" * (len(table_name) + 20) + "\n")

    except sqlite3.Error as e:
        print(f"❌ 数据库检查失败: {e}")


if __name__ == "__main__":
    inspect_database()
