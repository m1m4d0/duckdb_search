# DuckDB Hybrid Search

DuckDBを用いた日本語ドキュメント向けハイブリッド検索（VSS + FTS）デモです。PLaMo埋め込みモデルでセマンティック検索し、DuckDBのVSS/HNSWとFTS/BM25で高速検索します。

## 特徴

- **ハイブリッド検索**: ベクトル検索と全文検索を併用
- **高速類似検索**: DuckDB VSS拡張 + HNSWインデックス
- **BM25全文検索**: DuckDB FTS拡張によるキーワード検索
- **日本語最適化**: pfnet/plamo-embedding-1b を使用
- **一度のモデルロード**: 起動時にモデルを読み込んで以降の検索を高速化

## 必要要件

- Python 3.14以上（`pyproject.toml`に準拠）
- CUDA対応GPUがあると高速（CPUでも動作可能）

## セットアップ

1) 依存関係をインストール

```bash
uv sync
```

2) Markdownからデータベースを作成（FTS/HNSW込み）

```bash
uv run src/create_hybrid.py
```

- 入力: `docs/md_rag_fts/*.md`（YAMLフロントマター + `[FTS]...[/FTS]`付き）
- 出力: `docs/db/duckdb_search.duckdb`
- モデルキャッシュ: `models/plamo/` に自動保存

## 使い方

- 対話モード（推奨）

```bash
uv run src/search_interactive.py
```

  - 起動後に VSS か FTS を選び、クエリ/キーワードを入力

- スクリプトデモ（VSS/FTSの両方を実行）

```bash
uv run src/search_hybrid.py
```

## プロジェクト構成

```
duckdb-hybrid-search/
├── src/
│   ├── embedding_model.py      # PLaMo埋め込みモデルの遅延ロード
│   ├── create_hybrid.py        # Markdown -> DuckDB (FTS/HNSW付き)
│   ├── search_hybrid.py        # VSS/FTSデモスクリプト
│   └── search_interactive.py   # 対話型ハイブリッド検索
├── docs/
│   ├── db/                     # DuckDBデータベース出力先
│   └── md_rag_fts/             # 検索対象のMarkdown（FTSキーワード付き）
└── models/                     # モデルキャッシュ（自動生成）
```

## データ形式

```markdown
---
document_name: "ドキュメント名"
document_path: "/path/to/doc"
category: "カテゴリ"
tag: "タグ"
---
最初のチャンクの内容

[FTS]検索キーワード1 キーワード2 キーワード3[/FTS]
---
2番目のチャンクの内容

[FTS]キーワード4 キーワード5[/FTS]
---
```

- `---`で区切られた各ブロックが1チャンクとして登録され、ベクトル化されます
- `[FTS]...[/FTS]`内のキーワードがBM25検索用に抽出されます

## 技術スタック

- DuckDB (VSS拡張 / FTS拡張 + HNSW)
- PLaMo embeddings (pfnet/plamo-embedding-1b) + PyTorch
- BM25 (DuckDB FTS)

## ライセンス

MIT License
