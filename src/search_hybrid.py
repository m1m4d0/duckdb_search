import duckdb
import torch
import time

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "duckdb_search"

# モデルの取得（起動時に一度だけ）
print("=" * 80)
print("モデル読み込み開始...")
start_time = time.perf_counter()
v_model, v_tokenizer = get_model_and_tokenizer()
model_load_time = time.perf_counter() - start_time
print(f"✓ モデル読み込み完了: {model_load_time:.3f}秒")
print("=" * 80)

# 接続プール（再利用可能な接続）
_conn = None


def get_connection():
    """データベース接続を取得（再利用）"""
    global _conn
    if _conn is None:
        db_path = f"docs/db/{DB_NAME}.duckdb"
        try:
            # 既に拡張がインストールされている前提で読み取り専用で開く
            _conn = duckdb.connect(db_path, read_only=True)
            _conn.load_extension("vss")
            _conn.load_extension("fts")
        except Exception:
            # 拡張未インストールの場合のみ書き込み可能にして導入し、その後read-onlyで再接続
            install_conn = duckdb.connect(db_path)
            install_conn.install_extension("vss")
            install_conn.load_extension("vss")
            install_conn.install_extension("fts")
            install_conn.load_extension("fts")
            install_conn.close()

            _conn = duckdb.connect(db_path, read_only=True)
            _conn.load_extension("vss")
            _conn.load_extension("fts")
    return _conn


def vss_search(query, limit=10):
    """ベクトル類似度検索を実行"""
    total_start = time.perf_counter()

    # DB接続
    conn_start = time.perf_counter()
    conn = get_connection()
    conn_time = time.perf_counter() - conn_start
    print(f"  [DB接続] {conn_time:.3f}秒")

    with torch.inference_mode():
        # クエリ埋め込み生成
        embed_start = time.perf_counter()
        query_embedding = v_model.encode_query(query, v_tokenizer)
        embed_time = time.perf_counter() - embed_start
        print(f"  [埋め込み生成] {embed_time:.3f}秒")

        # HNSWインデックスを利用した高速検索
        search_start = time.perf_counter()
        rows = conn.sql(
            """
            SELECT
                id,
                array_cosine_distance(content_v, ?::FLOAT[2048]) as distance,
                document_name,
                document_path,
                category,
                tag,
                content
            FROM documents
            ORDER BY distance ASC
            LIMIT ?
            """,
            params=[query_embedding.cpu().squeeze().numpy().tolist(), limit],
        ).fetchall()
        search_time = time.perf_counter() - search_start
        print(f"  [ベクトル検索] {search_time:.3f}秒")

        total_time = time.perf_counter() - total_start
        print(f"  [合計] {total_time:.3f}秒")

        return rows


def fts_search(query, limit=10):
    """全文検索を実行（BM25スコアリング）"""
    total_start = time.perf_counter()

    # DB接続
    conn_start = time.perf_counter()
    conn = get_connection()
    conn_time = time.perf_counter() - conn_start
    print(f"  [DB接続] {conn_time:.3f}秒")

    # BM25検索
    search_start = time.perf_counter()
    rows = conn.sql(
        """
        SELECT
            id,
            document_name,
            document_path,
            category,
            tag,
            content,
            content_fts,
            score
        FROM (
            SELECT *, fts_main_documents.match_bm25(id, ?) AS score
            FROM documents
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
        """,
        params=[query, limit],
    ).fetchall()
    search_time = time.perf_counter() - search_start
    print(f"  [FTS検索(BM25)] {search_time:.3f}秒")

    total_time = time.perf_counter() - total_start
    print(f"  [合計] {total_time:.3f}秒")

    return rows


def search_vss_display(query, limit=3):
    """VSS検索を実行して結果を表示"""
    print(f"Query: {query}")
    print("=" * 80)
    print("\n【ベクトル類似度検索（VSS）上位{0}件】".format(limit))
    print("-" * 80)
    rows = vss_search(query, limit)
    for id, distance, document_name, document_path, category, tag, content in rows:
        similarity = 1 - distance
        print(f"ID: {id}, Similarity: {similarity:.4f}")
        print(f"  Document: {document_name}")
        print(f"  Category: {category}")
        print(f"  Tag: {tag}")
        print(
            f"  Content: {content[:100]}..."
            if len(content) > 100
            else f"  Content: {content}"
        )
        print()
    return rows


def search_fts_display(keywords, limit=3):
    """FTS検索を実行して結果を表示"""
    print(f"Keywords: {keywords}")
    print("=" * 80)
    print("\n【全文検索（FTS/BM25）上位{0}件】".format(limit))
    print("-" * 80)
    rows = fts_search(keywords, limit)
    if not rows:
        print("  (該当なし)")
    for (
        id,
        document_name,
        document_path,
        category,
        tag,
        content,
        content_fts,
        score,
    ) in rows:
        print(f"ID: {id}, BM25 Score: {score:.4f}")
        print(f"  Document: {document_name}")
        print(f"  Category: {category}")
        print(f"  Tag: {tag}")
        print(f"  FTS Keywords: {content_fts}")
        print(
            f"  Content: {content[:100]}..."
            if len(content) > 100
            else f"  Content: {content}"
        )
        print()
    return rows


def close_connection():
    """接続を明示的にクローズ（終了時に使用）"""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def main():
    query = "設備監視について教えてください"
    # クエリからキーワードを抽出（実際はLLM等で抽出する想定）
    keywords = "設備 監視"

    print("\n【VSS検索】")
    search_vss_display(query)

    print("\n" + "=" * 80)
    print("\n【FTS検索】")
    search_fts_display(keywords)

    # プログラム終了時
    close_connection()


if __name__ == "__main__":
    main()
