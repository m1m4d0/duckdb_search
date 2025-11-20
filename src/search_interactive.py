"""
対話型ベクトル検索デモ

モデルロードは起動時の1回のみ。
以降の検索は高速に実行できることを実感できます。
"""
import duckdb
import torch
import time

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "facility_assist"

print("=" * 80)
print("📚 ベクトル検索システム - 対話モード")
print("=" * 80)

# モデルの取得（起動時に一度だけ）
print("\n🔄 モデル読み込み中...")
print("  ※ この処理は起動時の1回のみです（サービス起動に相当）")
start_time = time.perf_counter()
v_model, v_tokenizer = get_model_and_tokenizer()
model_load_time = time.perf_counter() - start_time
print(f"✅ モデル読み込み完了: {model_load_time:.3f}秒")
print("\n" + "=" * 80)
print("準備完了！検索クエリを入力してください")
print("（'exit' または 'quit' で終了）")
print("=" * 80 + "\n")

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


def vss_search(query, limit=5):
    """ベクトル類似度検索を実行"""
    total_start = time.perf_counter()

    conn = get_connection()

    with torch.inference_mode():
        # クエリ埋め込み生成
        embed_start = time.perf_counter()
        query_embedding = v_model.encode_query(query, v_tokenizer)
        embed_time = time.perf_counter() - embed_start

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

        total_time = time.perf_counter() - total_start

        return rows, {
            "embed_time": embed_time,
            "search_time": search_time,
            "total_time": total_time,
        }


def display_results(query, rows, timings):
    """検索結果を表示"""
    print(f"\n🔍 検索: '{query}'")
    print(f"⏱️  処理時間: {timings['total_time']:.3f}秒 "
          f"(埋め込み: {timings['embed_time']:.3f}秒, 検索: {timings['search_time']:.3f}秒)")
    print("-" * 80)

    if not rows:
        print("❌ 結果が見つかりませんでした")
        return

    for idx, (id, distance, document_name, document_path, category, tag, content) in enumerate(rows, 1):
        similarity = 1 - distance  # cosine distance -> similarity
        print(f"\n[{idx}] ID: {id} | 類似度: {similarity:.4f}")
        print(f"    📄 {document_name} ({category})")

        # 内容を適切な長さで表示
        content_preview = content[:150].replace("\n", " ")
        if len(content) > 150:
            content_preview += "..."
        print(f"    💬 {content_preview}")


def close_connection():
    """接続を明示的にクローズ"""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def main():
    """対話型検索のメインループ"""
    search_count = 0

    try:
        while True:
            # ユーザー入力を取得
            try:
                query = input("\n検索> ").strip()
            except EOFError:
                print("\n👋 終了します")
                break

            # 終了コマンドのチェック
            if query.lower() in ["exit", "quit", "q", "終了"]:
                print("\n👋 終了します")
                break

            # 空入力のスキップ
            if not query:
                continue

            # 検索実行
            search_count += 1
            try:
                rows, timings = vss_search(query, limit=5)
                display_results(query, rows, timings)
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Ctrl+C で終了します")
    finally:
        # 統計情報を表示
        print("\n" + "=" * 80)
        print(f"📊 統計情報")
        print(f"  総検索回数: {search_count}回")
        print(f"  モデルロード: 1回のみ ({model_load_time:.3f}秒)")
        print("=" * 80)

        # 接続をクローズ
        close_connection()


if __name__ == "__main__":
    main()
