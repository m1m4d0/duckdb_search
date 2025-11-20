import duckdb
import torch
import time

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "facility_assist"

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
        _conn = duckdb.connect(db_path, read_only=True)
        _conn.install_extension("vss")
        _conn.load_extension("vss")
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


def search(query, limit=10):
    """検索を実行して結果を表示"""
    print(f"Query: {query}")
    print("-" * 80)

    rows = vss_search(query, limit)
    for id, distance, document_name, document_path, category, tag, content in rows:
        similarity = 1 - distance  # cosine distance -> similarity
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


def close_connection():
    """接続を明示的にクローズ（終了時に使用）"""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def main():
    print("\n【1回目の検索】（初回）")
    search("設備監視について教えてください")

    print("\n" + "=" * 80)
    print("【2回目の検索】（モデル読み込み済み）")
    search("利用料金について")

    # プログラム終了時
    close_connection()


if __name__ == "__main__":
    main()
