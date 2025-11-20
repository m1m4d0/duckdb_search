import torch
from transformers import AutoModel, AutoTokenizer
import os

# 設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルの初期化（モジュールロード時に一度だけ実行）
_model = None
_tokenizer = None


def get_model_and_tokenizer():
    """モデルとトークナイザーを取得（遅延初期化）"""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        model_path = "models/plamo"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            _tokenizer = AutoTokenizer.from_pretrained(
                "pfnet/plamo-embedding-1b", trust_remote_code=True
            )
            _model = AutoModel.from_pretrained(
                "pfnet/plamo-embedding-1b", trust_remote_code=True
            )
            _tokenizer.save_pretrained(model_path)
            _model.save_pretrained(model_path)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        _model = _model.to(device)

    return _model, _tokenizer
