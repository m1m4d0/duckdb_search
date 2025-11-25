# CLAUDE.md

日本語で回答してください。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

DuckDBを使用したハイブリッド検索システム（プロジェクト名: duckdb-hybrid-search）。日本語ドキュメント検索に最適化されており、以下の技術を使用:
- PLaMo埋め込みモデル (pfnet/plamo-embedding-1b) による高品質な日本語セマンティック検索
- DuckDBのVSS拡張とHNSWインデックスによる高速なベクトル類似度検索
- DuckDBのFTS拡張とBM25による全文検索
- PyTorchによるモデル推論とGPUアクセラレーション

## アーキテクチャ

### コアコンポーネント

**埋め込み層** (`src/embedding_model.py`)
- PLaMo埋め込みモデルの遅延初期化（初回使用時のみロード）
- モデルを`models/plamo/`ディレクトリにキャッシュ
- 全モジュールで共有するため`get_model_and_tokenizer()`を提供
- CUDA利用可能時は自動検出

**ハイブリッドデータベース初期化** (`src/create_hybrid.py`)
- `docs/md_rag_fts/`からYAMLフロントマター付きMarkdownファイルを解析
- `---`セパレータでドキュメントをチャンク分割
- `[FTS]...[/FTS]`タグからFTS用キーワードを抽出
- 各チャンクに対して2048次元の埋め込みベクトルを生成
- テーブルが存在しなければ新規作成、存在すれば追加モード
- **FTSインデックスとHNSWインデックスを両方自動作成**
- 永続化フラグ`hnsw_enable_experimental_persistence`を使用してHNSWインデックスを永続化

**ハイブリッド検索インターフェース** (`src/search_hybrid.py`)
- `get_connection()`による読み取り専用接続プーリング（VSS/FTS両拡張をロード）
- `vss_search()`: コサイン距離メトリックでHNSWインデックスを使用したベクトル検索
- `fts_search()`: BM25スコアリングによる全文検索
- `search_vss_display()`: VSS検索結果を表示
- `search_fts_display()`: FTS検索結果を表示
- デフォルトデータベース: `docs/db/duckdb_search.duckdb`
- VSS検索とFTS検索の処理時間を詳細に計測

**対話型検索** (`src/search_interactive.py`)
- 起動時にモデルを1回だけロード（サービス起動に相当）
- 対話モードで検索タイプを選択（1: VSS / 2: FTS）
- 以降の検索は高速実行（埋め込み生成とDB検索のみ）
- ユーザーフレンドリーな結果表示（絵文字による視覚的フィードバック）
- 検索統計情報の表示（処理時間、検索回数など）
- `exit`/`quit`コマンドで終了

## よく使うコマンド

### セットアップと開発

```bash
# 依存関係のインストール (Python 3.14以上が必要)
uv sync
```

### データベース操作

```bash
# Markdownファイルからハイブリッド検索用データベースを作成
# （FTSインデックスとHNSWインデックスを両方自動作成）
uv run src/create_hybrid.py
```

### 検索の実行

```bash
# 対話型ハイブリッド検索（推奨）
# VSS（ベクトル検索）またはFTS（全文検索）を対話的に選択可能
uv run src/search_interactive.py

# バッチ処理用のハイブリッド検索デモ
uv run src/search_hybrid.py
# ※ クエリを変更する場合はmain()内を編集
```

## データベーススキーマ

テーブル: `documents`
- `id` - 自動増分の主キー（シーケンスから生成）
- `document_name` - YAMLフロントマターから取得
- `document_path` - YAMLフロントマターから取得
- `category` - YAMLフロントマターから取得
- `tag` - YAMLフロントマターから取得
- `content` - チャンク分割されたテキストコンテンツ
- `content_fts` - FTS用キーワード（`[FTS]...[/FTS]`タグから抽出）
- `content_v` - FLOAT[2048]型の埋め込みベクトル
- `created_at` - タイムスタンプ

インデックス:
- `documents_vss_idx` - content_v上のHNSW（コサインメトリック）
- `fts_main_documents` - content上のFTSインデックス（BM25検索用）

## 重要な実装詳細

### モデル読み込みパターン
全モジュールは`get_model_and_tokenizer()`による遅延初期化を使用。モデルはグローバルにキャッシュされ、重複ロードを防ぐ。

### ドキュメントのチャンク分割
`docs/md_rag_fts/`内のMarkdownファイルは以下の形式を使用:
```
---
document_name: "名前"
document_path: "/パス"
category: "カテゴリ"
tag: "タグ"
---
最初のチャンクの内容

[FTS]キーワード1 キーワード2 キーワード3[/FTS]
---
2番目のチャンクの内容

[FTS]キーワード4 キーワード5[/FTS]
---
```

### インデックス管理ワークフロー
1. `create_hybrid.py`を実行してデータベースを作成
2. FTSインデックスとHNSWインデックスの両方が自動的に作成される
3. 永続化フラグ`hnsw_enable_experimental_persistence`により、HNSWインデックスがデータベースファイルに永続化される

### 接続の取り扱い
- `create_hybrid.py`: 書き込み操作、FTS/HNSWインデックス作成、完了後に接続をクローズ
- `search_hybrid.py`: 読み取り専用の接続プーリング、シャットダウン時に`close_connection()`を呼び出す
- `search_interactive.py`: 読み取り専用の接続プーリング、対話セッション中は接続を再利用、終了時に`close_connection()`を呼び出す

### パフォーマンス最適化
- **モデルロード**: 全スクリプトで`get_model_and_tokenizer()`による遅延初期化とキャッシュを使用
- **HNSWインデックス**: 線形スキャンではなくHNSWインデックスにより高速な近似最近傍探索を実現
- **接続プーリング**: 検索スクリプトは読み取り専用接続を再利用し、接続のオーバーヘッドを削減
- **バッチ処理**: 複数ドキュメントの処理時は`torch.inference_mode()`で推論を最適化
