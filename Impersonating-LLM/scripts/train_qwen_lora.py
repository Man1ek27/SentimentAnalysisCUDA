#!/usr/bin/env python3
"""
Trump Tweet Fine-Tuning: Qwen2.5-1.5B + LoRA
Server-ready script — no Colab dependencies.

Usage (basic):
    python train_qwen_lora.py

Usage (with local CSV, custom output):
    python train_qwen_lora.py --data-path /path/to/tweets.csv --output-dir ~/models/trump_qwen

Usage (generate only from saved adapter):
    python train_qwen_lora.py --generate-only ~/models/trump_qwen/best_adapter

Key speedups vs. naive training:
  - fp16 mixed precision (torch.cuda.amp)
  - Gradient accumulation (larger effective batch without OOM)
  - pin_memory + num_workers in DataLoader
  - torch.compile() on PyTorch >= 2.0
"""

import argparse
import math
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


BASE_MODEL = "Qwen/Qwen2.5-1.5B"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B on Trump tweets with LoRA")
    p.add_argument("--data-path", type=str, default=None,
                   help="Path to tweets CSV. If omitted, downloads via Kaggle CLI.")
    p.add_argument("--output-dir", type=str, default="./checkpoints/trump_qwen_lora",
                   help="Directory for saving the best LoRA adapter (default: ./checkpoints/trump_qwen_lora)")
    p.add_argument("--sample-size", type=int, default=10_000,
                   help="Number of tweets to sample from the dataset (default: 10000)")
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--grad-accum",  type=int,   default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum (default: 4 → eff. 32)")
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--max-length",  type=int,   default=128)
    p.add_argument("--lora-r",      type=int,   default=8)
    p.add_argument("--lora-alpha",  type=int,   default=16)
    p.add_argument("--no-compile",  action="store_true",
                   help="Disable torch.compile() (use on PyTorch < 2.0 or if it causes issues)")
    p.add_argument("--skip-baseline-ppl", action="store_true",
                   help="Skip loading base model for perplexity comparison (saves VRAM at end)")
    p.add_argument("--generate-only", type=str, default=None, metavar="ADAPTER_PATH",
                   help="Skip training — load adapter from this path and generate tweets")
    return p.parse_args()


# ─── DATA ─────────────────────────────────────────────────────────────────────

def _download_via_kaggle(dest: Path) -> Path:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "~/.kaggle/kaggle.json not found.\n"
            "Create it manually:\n"
            '  mkdir -p ~/.kaggle\n'
            '  echo \'{"username":"YOUR_USER","key":"YOUR_KEY"}\' > ~/.kaggle/kaggle.json\n'
            '  chmod 600 ~/.kaggle/kaggle.json\n'
            "Get credentials at https://www.kaggle.com/settings/account"
        )
    print("Downloading Trump's Legacy dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "zusmani/trumps-legacy", "-p", str(dest)],
        check=True,
    )
    zip_file = list(dest.glob("*.zip"))[0]
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(dest)
    return list(dest.glob("*.csv"))[0]


def load_tweets(data_path: str | None, sample_size: int) -> tuple[pd.DataFrame, list[str]]:
    if data_path:
        csv_path = Path(data_path)
    else:
        cache = Path("/tmp/trump_data")
        cache.mkdir(exist_ok=True)
        existing = list(cache.glob("*.csv"))
        csv_path = existing[0] if existing else _download_via_kaggle(cache)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    # Accept any column whose name contains "text" (case-insensitive), else first column
    text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
    df["text"] = df[text_col].astype(str).apply(_clean)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    print(f"After cleaning: {len(df)} tweets")

    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled: {len(df)} tweets")

    return df, df["text"].tolist()


def _clean(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── MEMORIZATION CHECK ───────────────────────────────────────────────────────

def _ngrams(text: str, n: int) -> set[tuple]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))


def check_memorization(generated: list[str], train_texts: list[str], n: int = 4) -> None:
    """
    For each generated tweet, compute what fraction of its 4-grams also appear
    in the training corpus. High overlap (>0.8) suggests memorization.
    """
    print(f"\n{'─'*60}")
    print(f"MEMORIZATION CHECK  ({n}-gram overlap with training data)")
    print(f"{'─'*60}")

    # Build training corpus n-gram index (use up to 5k tweets for speed)
    sample = train_texts[:5_000]
    train_ng: set[tuple] = set()
    for t in sample:
        train_ng.update(_ngrams(t, n))

    for i, gen in enumerate(generated, 1):
        gen_ng = _ngrams(gen, n)
        if not gen_ng:
            print(f"[{i}] (too short to check)")
            continue
        overlap = len(gen_ng & train_ng) / len(gen_ng)
        flag = "  ⚠  HIGH — possible memorization" if overlap > 0.8 else ""
        print(f"[{i}] overlap={overlap:.0%}{flag}")
        print(f"     {gen[:110]}")
    print()


# ─── MODEL ────────────────────────────────────────────────────────────────────

def load_base(device: torch.device) -> AutoModelForCausalLM:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": device},
    )


def apply_lora(model, lora_r: int, lora_alpha: int) -> object:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Qwen2.5 attention projection names
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(
    args: argparse.Namespace,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ── training pass ──
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        bar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{args.epochs} [train]")

        for step, (ids, mask) in bar:
            ids  = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                loss = model(ids, attention_mask=mask, labels=ids).loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * args.grad_accum
            bar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

        avg_train = train_loss / len(train_loader)

        # ── validation pass ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids, mask in val_loader:
                ids  = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    val_loss += model(ids, attention_mask=mask, labels=ids).loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save only the LoRA adapter weights (~10 MB), not full 3 GB model
            model.save_pretrained(out / "best_adapter")
            print(f"  -> Saved best adapter to {out / 'best_adapter'}  (val={avg_val:.4f})")

    return best_val_loss


# ─── GENERATION ───────────────────────────────────────────────────────────────

PROMPTS = [
    "Make America",
    "I will",
    "The fake news",
    "Beautiful",
    "Total disaster",
]


def generate_tweets(model, tokenizer, device: torch.device) -> list[str]:
    model.eval()
    results = []
    print("\nGenerated Trump-style tweets:\n")
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=60,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append(text)
        print(f'[{prompt!r}]\n  {text}\n')
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:  {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    # ── tokenizer (shared across all modes) ──
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── generate-only shortcut ──
    if args.generate_only:
        print(f"\nGenerate-only mode — loading adapter from {args.generate_only}")
        base = load_base(device)
        model = PeftModel.from_pretrained(base, args.generate_only)
        model.config.pad_token_id = tokenizer.eos_token_id
        _, train_texts = load_tweets(args.data_path, args.sample_size)
        generated = generate_tweets(model, tokenizer, device)
        check_memorization(generated, train_texts)
        return

    # ── data ──
    _, all_texts = load_tweets(args.data_path, args.sample_size)

    print("Tokenizing...")
    enc = tokenizer(
        all_texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

    nw = min(4, os.cpu_count() or 1)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
        num_workers=nw,
        prefetch_factor=2 if nw > 0 else None,
        persistent_workers=(nw > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # ── model + LoRA ──
    model = apply_lora(load_base(device), args.lora_r, args.lora_alpha)

    # torch.compile — significant speedup on CUDA if PyTorch >= 2.0
    if not args.no_compile and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile()  (first epoch will be slower)...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile() failed ({e}), continuing without it.")

    # ── training ──
    t0 = time.time()
    best_val_loss = train(args, model, train_loader, val_loader, device)
    print(f"\nTotal training time: {(time.time() - t0) / 60:.1f} min")

    # ── load best adapter and generate ──
    adapter_path = Path(args.output_dir) / "best_adapter"
    del model  # free VRAM before loading two models
    torch.cuda.empty_cache()

    base = load_base(device)
    ft_model = PeftModel.from_pretrained(base, adapter_path)
    ft_model.config.pad_token_id = tokenizer.eos_token_id

    generated = generate_tweets(ft_model, tokenizer, device)
    check_memorization(generated, all_texts)

    # ── perplexity comparison ──
    ft_ppl = math.exp(best_val_loss)
    print(f"Fine-tuned perplexity: {ft_ppl:.2f}")

    if not args.skip_baseline_ppl:
        del ft_model
        torch.cuda.empty_cache()
        orig = load_base(device)
        orig.eval()
        orig_loss = 0.0
        with torch.no_grad():
            for ids, mask in val_loader:
                ids  = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    orig_loss += orig(ids, attention_mask=mask, labels=ids).loss.item()
        orig_ppl = math.exp(orig_loss / len(val_loader))
        print(f"Base model perplexity:  {orig_ppl:.2f}")
        reduction = (orig_ppl / ft_ppl - 1) * 100
        print(f"Improvement: {orig_ppl - ft_ppl:.2f} pts  ({reduction:.1f}% reduction)")
    else:
        print("(baseline perplexity skipped — use --skip-baseline-ppl=false to enable)")


if __name__ == "__main__":
    main()
