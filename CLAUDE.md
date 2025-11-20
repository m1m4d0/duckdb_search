# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

DuckDBを使用した高速ベクトル検索システム（プロジェクト名: duckdb-hybrid-search）。日本語ドキュメント検索に最適化されており、以下の技術を使用:
- PLaMo埋め込みモデル (pfnet/plamo-embedding-1b) による高品質な日本語セマンティック検索
- DuckDBのVSS拡張とHNSWインデックスによる高速な類似度検索
- PyTorchによるモデル推論とGPUアクセラレーション

## アーキテクチャ

### コアコンポーネント

**埋め込み層** (`src/embedding_model.py`)
- PLaMo埋め込みモデルの遅延初期化（初回使用時のみロード）
- モデルを`models/plamo/`ディレクトリにキャッシュ
- 全モジュールで共有するため`get_model_and_tokenizer()`を提供
- CUDA利用可能時は自動検出

**データベース初期化** (`src/create_vss.py`)
- `docs/md_rag/`からYAMLフロントマター付きMarkdownファイルを解析
- `---`セパレータでドキュメントをチャンク分割
- 各チャンクに対して2048次元の埋め込みベクトルを生成
- 2つのモード: MODE=1 (テーブル再作成), MODE=2 (既存テーブルに追加)
- **重要**: HNSWインデックスは自動作成されない - 別途作成が必要

**インデックス管理** (`src/index_manager.py`)
- ベクトル検索用のHNSWインデックスを管理
- CLIツール、コマンド: create, drop, info, ensure
- 実験的な永続化機能は`hnsw_enable_experimental_persistence`で有効化
- Web API起動時は`ensure_index()`でインデックスの存在を保証

**検索インターフェース** (`src/search_vss.py`)
- `get_connection()`による読み取り専用接続プーリング
- コサイン距離メトリックでHNSWインデックスを使用
- 類似度スコア付きでランク付けされた結果を返す
- デフォルトデータベース: `docs/db/facility_assist.duckdb`
- `search()`関数を他のモジュールから利用可能

**対話型検索** (`src/search_interactive.py`)
- 起動時にモデルを1回だけロード（サービス起動に相当）
- 以降の検索は高速実行（埋め込み生成とDB検索のみ）
- ユーザーフレンドリーな結果表示
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
# Markdownファイルから埋め込みベクトル付きデータベースを作成
uv run src/create_vss.py

# HNSWインデックスの作成（データベース作成後に必須）
uv run src/index_manager.py --db docs/db/facility_assist.duckdb ensure

# インデックス管理
uv run src/index_manager.py --db <パス> create [--force]
uv run src/index_manager.py --db <パス> info --index documents_vss_idx
uv run src/index_manager.py --db <パス> drop --index documents_vss_idx
```

### 検索の実行

```bash
# 対話型ベクトル検索（推奨）
uv run src/search_interactive.py

# バッチ処理用のベクトル検索
uv run src/search_vss.py
# ※ クエリを変更する場合はmain()内を編集するか、search()関数をインポート
```

## データベーススキーマ

テーブル: `documents`
- `id` - 自動増分の主キー（シーケンスから生成）
- `document_name` - YAMLフロントマターから取得
- `document_path` - YAMLフロントマターから取得
- `category` - YAMLフロントマターから取得
- `tag` - YAMLフロントマターから取得
- `content` - チャンク分割されたテキストコンテンツ
- `content_v` - FLOAT[2048]型の埋め込みベクトル
- `created_at` - タイムスタンプ

インデックス: `documents_vss_idx` (content_v上のHNSW、コサインメトリック)

## 重要な実装詳細

### モデル読み込みパターン
全モジュールは`get_model_and_tokenizer()`による遅延初期化を使用。モデルはグローバルにキャッシュされ、重複ロードを防ぐ。

### ドキュメントのチャンク分割
`docs/md_rag/`内のMarkdownファイルは以下の形式を使用:
```
---
document_name: "名前"
document_path: "/パス"
category: "カテゴリ"
tag: "タグ"
---
最初のチャンクの内容
---
2番目のチャンクの内容
---
```

### インデックス管理ワークフロー
1. `create_vss.py`を実行してデータベースを作成（インデックスは作成されない）
2. `index_manager.py ensure`を実行してHNSWインデックスを作成
3. Web API用途では、起動時に`IndexManager.ensure_index()`を呼び出す

### 接続の取り扱い
- `create_vss.py`: 書き込み操作、完了後に接続をクローズ
- `search_vss.py`: 読み取り専用の接続プーリング、シャットダウン時に`close_connection()`を呼び出す
- `search_interactive.py`: 読み取り専用の接続プーリング、対話セッション中は接続を再利用、終了時に`close_connection()`を呼び出す
- `index_manager.py`: 操作ごとに短命な接続を使用

### パフォーマンス最適化
- **モデルロード**: 全スクリプトで`get_model_and_tokenizer()`による遅延初期化とキャッシュを使用
- **HNSWインデックス**: 線形スキャンではなくHNSWインデックスにより高速な近似最近傍探索を実現
- **接続プーリング**: 検索スクリプトは読み取り専用接続を再利用し、接続のオーバーヘッドを削減
- **バッチ処理**: 複数ドキュメントの処理時は`torch.inference_mode()`で推論を最適化
