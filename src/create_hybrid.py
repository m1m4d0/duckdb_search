import duckdb
import re
import torch
import yaml
from pathlib import Path

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "duckdb_search"
MD_DIR = "docs/md_rag_fts"

# モデルの取得
v_model, v_tokenizer = get_model_and_tokenizer()


def extract_fts_keywords(chunk: str) -> tuple[str, str]:
    """
    チャンクから[FTS]...[/FTS]タグを抽出し、本文とFTSキーワードを分離

    Returns:
        (本文, FTSキーワード) のタプル
    """
    fts_pattern = r"\[FTS\](.*?)\[/FTS\]"
    match = re.search(fts_pattern, chunk, re.DOTALL)

    if match:
        fts_keywords = match.group(1).strip()
        # FTSタグを除去した本文
        content_without_fts = re.sub(fts_pattern, "", chunk, flags=re.DOTALL).strip()
        return content_without_fts, fts_keywords
    else:
        # FTSタグがない場合は空文字列を返す
        return chunk.strip(), ""


def parse_markdown(md_path: str) -> tuple[dict, list[tuple[str, str]]]:
    """
    マークダウンファイルを解析し、フロントYAMLとチャンクリストを返す

    Returns:
        (front_yaml, [(本文, FTSキーワード), ...])
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ---で分割
    parts = content.split("---")

    # フロントYAML（1番目と2番目の---の間）
    if len(parts) >= 3:
        front_yaml = yaml.safe_load(parts[1].strip()) or {}
        # 2ブロック目以降をチャンクとして取得（---で区切られた各ブロック）
        raw_chunks = [part.strip() for part in parts[2:] if part.strip()]
        # 各チャンクからFTSキーワードを抽出
        chunks = [extract_fts_keywords(chunk) for chunk in raw_chunks]
    else:
        front_yaml = {}
        chunks = [extract_fts_keywords(content.strip())]

    return front_yaml, chunks


def main():
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
    conn.install_extension("fts")
    conn.load_extension("fts")

    # テーブルの存在確認
    result = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'documents'
    """).fetchone()
    table_exists = result[0] > 0 if result else False

    if table_exists:
        # テーブルが存在する場合は追加モード
        print("既存のテーブルにデータを追加します")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
    else:
        # テーブルが存在しない場合は新規作成
        print("テーブルを新規作成します")
        conn.execute("CREATE SEQUENCE id_sequence START 1;")
        conn.execute("""
            CREATE TABLE documents (
                id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY,
                document_name VARCHAR,
                document_path VARCHAR,
                category VARCHAR,
                tag VARCHAR,
                content VARCHAR,
                content_fts VARCHAR,
                content_v FLOAT[2048],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

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
            for i, (content, fts_keywords) in enumerate(chunks):
                # FTSタグを除いた本文をベクトル化
                content_embedding = v_model.encode_document([content], v_tokenizer)[0]
                print(f"  チャンク {i + 1}/{len(chunks)}: {content[:50]}...")

                conn.execute(
                    """
                    INSERT INTO documents (document_name, document_path, category, tag, content, content_fts, content_v)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        document_name,
                        document_path,
                        category,
                        tag,
                        content,
                        fts_keywords,
                        content_embedding.cpu().squeeze().numpy().tolist(),
                    ],
                )

    # FTSインデックスの作成（contentカラムに対して）
    print("\nFTSインデックスを作成中...")
    try:
        conn.execute("PRAGMA drop_fts_index('documents')")
    except Exception:
        pass  # インデックスが存在しない場合は無視
    conn.execute("""
        PRAGMA create_fts_index(
            'documents',
            'id',
            'content',
            stemmer='none',
            stopwords='none',
            ignore='',
            strip_accents=0,
            lower=0,
            overwrite=1
        )
    """)
    print("✓ FTSインデックス作成完了")

    # HNSWインデックスの作成（永続化を有効にして）
    print("\nHNSWインデックスを作成中...")
    conn.execute("SET hnsw_enable_experimental_persistence = true")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS documents_vss_idx
        ON documents
        USING HNSW (content_v)
        WITH (metric = 'cosine')
    """)
    print("✓ HNSWインデックス作成完了")

    conn.close()
    print(f"\nデータベース作成完了: {db_path}")


if __name__ == "__main__":
    main()
