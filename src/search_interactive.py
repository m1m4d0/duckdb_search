"""
対話型ハイブリッド検索デモ

モデルロードは起動時の1回のみ。
VSS（ベクトル検索）とFTS（全文検索）を選択して検索できます。
"""

import duckdb
import torch
import time

from embedding_model import get_model_and_tokenizer

# 設定
DB_NAME = "duckdb_search"

print("=" * 80)
print("📚 ハイブリッド検索システム - 対話モード")
print("=" * 80)

# モデルの取得（起動時に一度だけ）
print("\n🔄 モデル読み込み中...")
print("  ※ この処理は起動時の1回のみです（サービス起動に相当）")
start_time = time.perf_counter()
v_model, v_tokenizer = get_model_and_tokenizer()
model_load_time = time.perf_counter() - start_time
print(f"✅ モデル読み込み完了: {model_load_time:.3f}秒")
print("\n" + "=" * 80)
print("準備完了！検索モードを選択してクエリを入力してください")
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
        _conn.install_extension("fts")
        _conn.load_extension("fts")
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


def fts_search(keywords, limit=5):
    """全文検索を実行（BM25スコアリング）"""
    total_start = time.perf_counter()

    conn = get_connection()

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
        params=[keywords, limit],
    ).fetchall()
    search_time = time.perf_counter() - search_start

    total_time = time.perf_counter() - total_start

    return rows, {
        "search_time": search_time,
        "total_time": total_time,
    }


def display_vss_results(query, rows, timings):
    """VSS検索結果を表示"""
    print(f"\n🔍 VSS検索: '{query}'")
    print(
        f"⏱️  処理時間: {timings['total_time']:.3f}秒 "
        f"(埋め込み: {timings['embed_time']:.3f}秒, 検索: {timings['search_time']:.3f}秒)"
    )
    print("-" * 80)

    if not rows:
        print("❌ 結果が見つかりませんでした")
        return

    for idx, (
        id,
        distance,
        document_name,
        document_path,
        category,
        tag,
        content,
    ) in enumerate(rows, 1):
        similarity = 1 - distance  # cosine distance -> similarity
        print(f"\n[{idx}] ID: {id} | 類似度: {similarity:.4f}")
        print(f"    📄 {document_name} ({category})")

        # 内容を適切な長さで表示
        content_preview = content[:150].replace("\n", " ")
        if len(content) > 150:
            content_preview += "..."
        print(f"    💬 {content_preview}")


def display_fts_results(keywords, rows, timings):
    """FTS検索結果を表示"""
    print(f"\n🔍 FTS検索: '{keywords}'")
    print(f"⏱️  処理時間: {timings['total_time']:.3f}秒 (検索: {timings['search_time']:.3f}秒)")
    print("-" * 80)

    if not rows:
        print("❌ 結果が見つかりませんでした")
        return

    for idx, (
        id,
        document_name,
        document_path,
        category,
        tag,
        content,
        content_fts,
        score,
    ) in enumerate(rows, 1):
        print(f"\n[{idx}] ID: {id} | BM25スコア: {score:.4f}")
        print(f"    📄 {document_name} ({category})")
        if len(content_fts) > 80:
            print(f"    🏷️  FTSキーワード: {content_fts[:80]}...")
        else:
            print(f"    🏷️  FTSキーワード: {content_fts}")

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
            # 検索モード選択
            print("\n検索モードを選択してください:")
            print("  1: VSS（ベクトル類似度検索）")
            print("  2: FTS（全文検索/BM25）")
            try:
                mode = input("モード (1/2)> ").strip()
            except EOFError:
                print("\n👋 終了します")
                break

            if mode.lower() in ["exit", "quit", "q", "終了"]:
                print("\n👋 終了します")
                break

            if mode not in ["1", "2"]:
                print("⚠️  1 または 2 を入力してください")
                continue

            # クエリ入力
            try:
                if mode == "1":
                    query = input("検索クエリ> ").strip()
                else:
                    query = input("検索キーワード（スペース区切り）> ").strip()
            except EOFError:
                print("\n👋 終了します")
                break

            if query.lower() in ["exit", "quit", "q", "終了"]:
                print("\n👋 終了します")
                break

            if not query:
                continue

            # 検索実行
            search_count += 1
            try:
                if mode == "1":
                    rows, timings = vss_search(query, limit=5)
                    display_vss_results(query, rows, timings)
                else:
                    rows, timings = fts_search(query, limit=5)
                    display_fts_results(query, rows, timings)
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Ctrl+C で終了します")
    finally:
        # 統計情報を表示
        print("\n" + "=" * 80)
        print("📊 統計情報")
        print(f"  総検索回数: {search_count}回")
        print(f"  モデルロード: 1回のみ ({model_load_time:.3f}秒)")
        print("=" * 80)

        # 接続をクローズ
        close_connection()


if __name__ == "__main__":
    main()
