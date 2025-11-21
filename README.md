# DuckDB Hybrid Search

DuckDBを使用した高速ベクトル検索システム。日本語ドキュメント検索に最適化されたセマンティック検索を提供します。

## 特徴

- **ハイブリッド検索**: ベクトル検索（VSS）と全文検索（FTS/BM25）の両方をサポート
- **高速ベクトル検索**: DuckDBのVSS拡張とHNSWインデックスによる高速な類似度検索
- **BM25全文検索**: DuckDBのFTS拡張によるキーワードベースの検索
- **日本語最適化**: PLaMo埋め込みモデル（pfnet/plamo-embedding-1b）による高品質な日本語セマンティック検索
- **対話型インターフェース**: モデルロードは起動時の1回のみ、以降は高速レスポンス
- **シンプルなアーキテクチャ**: DuckDB単体で完結、追加のベクトルDBサーバー不要

## 必要要件

- Python 3.14以上
- CUDA対応GPU（推奨、CPUでも動作可能）

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. データベースの作成

Markdownファイルから埋め込みベクトル付きデータベースを作成：

```bash
uv run src/create_vss.py
```

### 3. HNSWインデックスの作成

データベース作成後、検索の高速化のためにインデックスを作成：

```bash
uv run src/index_manager.py --db docs/db/duckdb_search.duckdb ensure
```

## 使い方

### 対話型検索（推奨）

モデルを一度読み込んだ後、複数の検索を高速に実行できます：

```bash
uv run src/search_interactive.py
```

起動後、検索クエリを入力するだけで類似ドキュメントを検索できます。`exit`または`quit`で終了します。

### スクリプトからの利用

```python
from src.search_vss import vss_search, fts_search

# ベクトル類似度検索（自然文クエリ）
vss_results = vss_search("設備監視について教えてください", limit=5)

# 全文検索（キーワード）
fts_results = fts_search("設備 監視", limit=5)
```

## プロジェクト構成

```
duckdb-hybrid-search/
├── src/
│   ├── embedding_model.py      # PLaMo埋め込みモデル管理
│   ├── create_vss.py           # データベース作成
│   ├── create_hybrid.py        # ハイブリッド検索用データベース作成
│   ├── index_manager.py        # HNSWインデックス管理
│   ├── search_vss.py           # 検索API
│   ├── search_hybrid.py        # ハイブリッド検索API
│   └── search_interactive.py   # 対話型検索インターフェース
├── docs/
│   ├── db/                     # DuckDBデータベース
│   └── md_rag_fts/             # 検索対象のMarkdownファイル（FTSキーワード付き）
└── models/                     # モデルキャッシュディレクトリ（自動生成）
```

## データ形式

Markdownファイルは以下の形式で配置します：

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

- 各チャンクは独立したドキュメントとして埋め込みベクトルが生成され、VSS検索対象になります
- `[FTS]...[/FTS]`タグ内のキーワードはFTS/BM25検索用に抽出されます

## 技術スタック

- **DuckDB**: 高速な分析型データベース
  - VSS拡張: ベクトル検索をサポート
  - FTS拡張: BM25による全文検索をサポート
- **PLaMo**: Preferred Networksの日本語特化埋め込みモデル（2048次元）
- **PyTorch**: モデル推論
- **HNSW**: 高速な近似最近傍探索アルゴリズム
- **BM25**: 情報検索の標準的なランキングアルゴリズム

## ライセンス

MIT License

# 今後の展望

【入力】ユーザーの質問
       ├─► A. VSS向け LLMクエリ正規化（意味補完）
       │          例：抽象化・意図抽出・要約
       │
       └─► B. FTS向け クエリ表層正規化（誤字補正＋最小限の整形）
                  例：単語は変えない、固有名詞・数字は保護
                  
【検索】
1) Aクエリで VSS top-K
2) Bクエリで FTS(BM25) top-K
3) 両者をマージし rerank（Reciprocal Rank Fusion など）
4) LLM に渡す
