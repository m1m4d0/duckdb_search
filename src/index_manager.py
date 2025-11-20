"""
DuckDB HNSWインデックス管理モジュール

Web API起動時やメンテナンス時にインデックスを管理するためのユーティリティ
"""

import duckdb
from pathlib import Path
from typing import Optional


class IndexManager:
    """HNSWインデックスの管理クラス"""

    def __init__(self, db_path: str, enable_persistence: bool = True):
        """
        Args:
            db_path: データベースファイルのパス
            enable_persistence: 実験的な永続化機能を有効にするか
        """
        self.db_path = db_path
        self.enable_persistence = enable_persistence

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """データベース接続を取得"""
        conn = duckdb.connect(self.db_path)
        conn.install_extension("vss")
        conn.load_extension("vss")

        if self.enable_persistence:
            conn.execute("SET hnsw_enable_experimental_persistence = true")

        return conn

    def index_exists(self, table_name: str = "documents", index_name: str = "documents_vss_idx") -> bool:
        """
        インデックスが存在するかチェック

        Args:
            table_name: テーブル名
            index_name: インデックス名

        Returns:
            インデックスが存在する場合True
        """
        conn = self._get_connection()
        try:
            result = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM duckdb_indexes()
                WHERE index_name = ?
                """,
                [index_name],
            ).fetchone()
            return result[0] > 0 if result else False
        finally:
            conn.close()

    def create_index(
        self,
        table_name: str = "documents",
        column_name: str = "content_v",
        index_name: str = "documents_vss_idx",
        metric: str = "cosine",
        force_recreate: bool = False,
    ) -> dict:
        """
        HNSWインデックスを作成

        Args:
            table_name: テーブル名
            column_name: ベクトルカラム名
            index_name: インデックス名
            metric: 距離メトリック ('cosine', 'l2', 'ip')
            force_recreate: 既存インデックスを強制的に再作成

        Returns:
            作成結果の情報を含む辞書
        """
        conn = self._get_connection()
        try:
            # インデックスの存在確認
            if self.index_exists(table_name, index_name):
                if force_recreate:
                    print(f"既存のインデックス '{index_name}' を削除します...")
                    conn.execute(f"DROP INDEX IF EXISTS {index_name};")
                else:
                    return {
                        "status": "skipped",
                        "message": f"インデックス '{index_name}' は既に存在します",
                        "index_name": index_name,
                    }

            # インデックス作成
            print(f"HNSWインデックス '{index_name}' を作成中...")
            conn.execute(f"""
                CREATE INDEX {index_name}
                ON {table_name}
                USING HNSW ({column_name})
                WITH (metric = '{metric}');
            """)

            # 作成確認
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            return {
                "status": "created",
                "message": f"インデックス '{index_name}' を作成しました",
                "index_name": index_name,
                "table_name": table_name,
                "row_count": count,
                "metric": metric,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"インデックス作成エラー: {str(e)}",
                "error": str(e),
            }
        finally:
            conn.close()

    def drop_index(self, index_name: str = "documents_vss_idx") -> dict:
        """
        インデックスを削除

        Args:
            index_name: インデックス名

        Returns:
            削除結果の情報を含む辞書
        """
        conn = self._get_connection()
        try:
            if not self.index_exists(index_name=index_name):
                return {
                    "status": "not_found",
                    "message": f"インデックス '{index_name}' は存在しません",
                }

            conn.execute(f"DROP INDEX {index_name};")

            return {
                "status": "dropped",
                "message": f"インデックス '{index_name}' を削除しました",
                "index_name": index_name,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"インデックス削除エラー: {str(e)}",
                "error": str(e),
            }
        finally:
            conn.close()

    def get_index_info(self, index_name: str = "documents_vss_idx") -> Optional[dict]:
        """
        インデックスの情報を取得

        Args:
            index_name: インデックス名

        Returns:
            インデックス情報を含む辞書、存在しない場合はNone
        """
        conn = self._get_connection()
        try:
            result = conn.execute(
                """
                SELECT
                    index_name,
                    table_name,
                    sql
                FROM duckdb_indexes()
                WHERE index_name = ?
                """,
                [index_name],
            ).fetchone()

            if result:
                return {
                    "index_name": result[0],
                    "table_name": result[1],
                    "sql": result[2],
                    "exists": True,
                }
            return None

        finally:
            conn.close()

    def ensure_index(
        self,
        table_name: str = "documents",
        column_name: str = "content_v",
        index_name: str = "documents_vss_idx",
        metric: str = "cosine",
    ) -> dict:
        """
        インデックスが存在することを保証（なければ作成）

        Web API起動時などに使用する便利メソッド

        Args:
            table_name: テーブル名
            column_name: ベクトルカラム名
            index_name: インデックス名
            metric: 距離メトリック

        Returns:
            処理結果の情報を含む辞書
        """
        if self.index_exists(table_name, index_name):
            info = self.get_index_info(index_name)
            return {
                "status": "exists",
                "message": f"インデックス '{index_name}' は既に存在します",
                "info": info,
            }
        else:
            return self.create_index(table_name, column_name, index_name, metric)


def main():
    """スタンドアロンでインデックスを管理するためのCLI"""
    import argparse

    parser = argparse.ArgumentParser(description="DuckDB HNSWインデックス管理ツール")
    parser.add_argument(
        "--db", default="docs/db/facility_assist.duckdb", help="データベースファイルパス"
    )
    parser.add_argument(
        "command",
        choices=["create", "drop", "info", "ensure"],
        help="実行するコマンド",
    )
    parser.add_argument("--index", default="documents_vss_idx", help="インデックス名")
    parser.add_argument("--table", default="documents", help="テーブル名")
    parser.add_argument("--column", default="content_v", help="ベクトルカラム名")
    parser.add_argument("--metric", default="cosine", help="距離メトリック")
    parser.add_argument("--force", action="store_true", help="強制的に再作成")

    args = parser.parse_args()

    manager = IndexManager(args.db)

    if args.command == "create":
        result = manager.create_index(
            args.table, args.column, args.index, args.metric, args.force
        )
    elif args.command == "drop":
        result = manager.drop_index(args.index)
    elif args.command == "info":
        result = manager.get_index_info(args.index)
        if result is None:
            result = {"status": "not_found", "message": f"インデックス '{args.index}' は存在しません"}
    elif args.command == "ensure":
        result = manager.ensure_index(args.table, args.column, args.index, args.metric)

    print(f"\n結果:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
