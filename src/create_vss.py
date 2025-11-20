import duckdb
import torch
import yaml
from pathlib import Path

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "duckdb-search"
MD_DIR = "docs/md_rag"

# モデルの取得
v_model, v_tokenizer = get_model_and_tokenizer()


def parse_markdown(md_path: str) -> tuple[dict, list[str]]:
    """マークダウンファイルを解析し、フロントYAMLとチャンクリストを返す"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ---で分割
    parts = content.split("---")

    # フロントYAML（1番目と2番目の---の間）
    if len(parts) >= 3:
        front_yaml = yaml.safe_load(parts[1].strip()) or {}
        # 2ブロック目以降をチャンクとして取得（---で区切られた各ブロック）
        chunks = [part.strip() for part in parts[2:] if part.strip()]
    else:
        front_yaml = {}
        chunks = [content.strip()]

    return front_yaml, chunks


def main():
    # 実行モードの選択（ハードコーディング）
    # 1: テーブルの再作成（既存データを削除して新規作成）
    # 2: 追加（既存データに追加）
    MODE = 1  # ここを1か2に変更してください

    # マークダウンファイルの読み込み
    md_dir = Path(MD_DIR)
    md_files = list(md_dir.glob("*.md"))
    print(f"処理するマークダウンファイル数: {len(md_files)}")

    # データベースディレクトリの作成
    db_dir = Path("docs/db")
    db_dir.mkdir(parents=True, exist_ok=True)

    # DuckDBの初期化
    db_path = f"docs/db/{DB_NAME}.duckdb"
    conn = duckdb.connect(db_path)
    conn.install_extension("vss")
    conn.load_extension("vss")

    if MODE == 1:
        # モード1: テーブルの再作成
        print("モード1: テーブルを再作成します（既存データを削除）")
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.execute("DROP SEQUENCE IF EXISTS id_sequence;")
        conn.execute("CREATE SEQUENCE id_sequence START 1;")
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                document_name VARCHAR,
                document_path VARCHAR,
                category VARCHAR,
                tag VARCHAR,
                content VARCHAR,
                content_v FLOAT[2048],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    elif MODE == 2:
        # モード2: 既存テーブルに追加
        print("モード2: 既存テーブルにデータを追加します")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                document_name VARCHAR,
                document_path VARCHAR,
                category VARCHAR,
                tag VARCHAR,
                content VARCHAR,
                content_v FLOAT[2048],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    else:
        conn.close()
        raise ValueError("MODEは1または2を指定してください")

    # データの挿入
    with torch.inference_mode():
        for md_file in md_files:
            print(f"処理中: {md_file.name}")
            front_yaml, chunks = parse_markdown(str(md_file))

            # フロントYAMLからメタデータを取得
            document_name = front_yaml.get("document_name", "")
            document_path = front_yaml.get("document_path", "")
            category = front_yaml.get("category", "")
            tag = front_yaml.get("tag", "")

            # 各チャンクをベクトル化して挿入
            for i, chunk in enumerate(chunks):
                content_embedding = v_model.encode_document([chunk], v_tokenizer)[0]
                print(f"  チャンク {i + 1}/{len(chunks)}: {chunk[:50]}...")

                conn.execute(
                    """
                    INSERT INTO documents (document_name, document_path, category, tag, content, content_v)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        document_name,
                        document_path,
                        category,
                        tag,
                        chunk,
                        content_embedding.cpu().squeeze().numpy().tolist(),
                    ],
                )

    conn.close()
    print(f"データベース作成完了: {db_path}")
    print("\n注意: HNSWインデックスは作成されていません")
    print("インデックスを作成するには以下のコマンドを実行してください:")
    print(f"  python src/index_manager.py --db {db_path} ensure")


if __name__ == "__main__":
    main()
