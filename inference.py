#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py  —  Load a trained Eng→Spa Transformer and translate sentences.

Usage:
    # Interactive mode
    python inference.py

    # Single sentence
    python inference.py "Hello, how are you?"

    # Point to a custom model directory
    python inference.py --model_dir /path/to/models/translation "I love pizza."
"""

import os
import sys
import json
import string
import argparse

# ---------------------------------------------------------------------------
# Make the local rui package importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rui.utils import TextVectorizer
from rui.torch.transformer import Transformer

# ---------------------------------------------------------------------------
# Default model directory (relative to this file)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "translation"
)

# ---------------------------------------------------------------------------
# Standardisation — MUST match train.py exactly
# ---------------------------------------------------------------------------
_STRIP_CHARS = string.punctuation + "¿"
_STRIP_CHARS = _STRIP_CHARS.replace("[", "").replace("]", "")

def custom_standardization(text: str) -> str:
    text = text.lower()
    for ch in _STRIP_CHARS:
        text = text.replace(ch, "")
    return text


# ---------------------------------------------------------------------------
# Load model + vectorizers
# ---------------------------------------------------------------------------
def load_model(model_dir: str, device: torch.device):
    """Load all saved artifacts and return (model, source_vec, target_vec, config)."""

    # 1. Config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        cfg = json.load(f)

    # 2. Vectorizers
    src_vec = TextVectorizer.load(
        os.path.join(model_dir, "source_vectorizer.pkl"),
        standardize=custom_standardization,
    )
    tgt_vec = TextVectorizer.load(
        os.path.join(model_dir, "target_vectorizer.pkl"),
        standardize=custom_standardization,
    )

    # 3. Model skeleton
    model = Transformer(
        n_layers=cfg["n_layers"],
        d_emb=cfg["d_emb"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        seq_len=cfg["seq_len"],
        dropout=0.0,   # no dropout at inference time
    )

    # 4. Weights
    weights_path = os.path.join(model_dir, "transformer.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"transformer.pt not found in {model_dir}")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, src_vec, tgt_vec, cfg


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------
def translate(sentence: str,
              model: Transformer,
              src_vec: TextVectorizer,
              tgt_vec: TextVectorizer,
              cfg: dict,
              device: torch.device) -> str:
    """Translate a single English sentence to Spanish using greedy decoding."""

    seq_len   = cfg["seq_len"]
    start_idx = cfg["start_idx"]
    end_idx   = cfg["end_idx"]

    # Encode source
    src_ids = torch.tensor(src_vec(sentence), dtype=torch.long).to(device)  # (1, seq_len)

    # Greedy decode: build target token-by-token
    decoded = [start_idx]

    with torch.no_grad():
        for _ in range(seq_len):
            # Pad / truncate target so far to seq_len
            tgt_ids = decoded[-seq_len:]                           # keep last seq_len tokens
            tgt_ids = tgt_ids + [0] * (seq_len - len(tgt_ids))    # right-pad with 0
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)  # (1, seq_len)

            logits = model((src_ids, tgt_tensor))  # (1, seq_len, tgt_vocab_size)

            # Take prediction at the position of the last real token
            pos        = min(len(decoded) - 1, seq_len - 1)
            next_token = torch.argmax(logits[0, pos, :]).item()

            if next_token == end_idx or next_token == 0:
                break
            decoded.append(next_token)

    # Convert token ids → words, skip [start]
    idx_to_word = tgt_vec.idx_to_word
    words = [idx_to_word.get(idx, "") for idx in decoded[1:]]
    words = [w for w in words if w and w not in ("[end]", "[start]", "")]
    return " ".join(words) if words else "(no translation)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="English → Spanish Transformer Translator"
    )
    parser.add_argument(
        "sentence", nargs="?", default=None,
        help="English sentence to translate (omit for interactive mode)",
    )
    parser.add_argument(
        "--model_dir", default=DEFAULT_MODEL_DIR,
        help=f"Path to the model directory (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {args.model_dir}")
    model, src_vec, tgt_vec, cfg = load_model(args.model_dir, device)
    print("Model loaded. Ready to translate.\n")

    if args.sentence:
        result = translate(args.sentence, model, src_vec, tgt_vec, cfg, device)
        print(f"English : {args.sentence}")
        print(f"Spanish : {result}")
    else:
        print("Enter an English sentence (or 'quit' to exit):\n")
        while True:
            try:
                sentence = input("English > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not sentence:
                continue
            if sentence.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            result = translate(sentence, model, src_vec, tgt_vec, cfg, device)
            print(f"Spanish > {result}\n")


if __name__ == "__main__":
    main()
