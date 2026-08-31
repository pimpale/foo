"""Convert calibration-zoo stratos traces to TRL prompt/completion format.

Reads {prompt, system, thinking, text, split} records, renders the prompt through
the student model's chat template, and emits <think>-formatted completions.
The source `system` prompt is dropped: it requests <|begin_of_thought|> markers
that were already stripped from the traces, and <think> is Qwen's native format.

Rows whose prompt+completion exceed --max-length tokens are dropped (not
truncated — a trace cut off mid-thought teaches the model to stop reasoning
without concluding).
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path.home() / "calibration-zoo-stratos/stratos_r1.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("json", data_files=args.input, split="train")

    def convert(rec):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": rec["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = f"<think>\n{rec['thinking']}\n</think>\n\n{rec['text']}"
        n_tokens = len(tok(prompt + completion).input_ids) + 1  # +1 for the EOS TRL appends
        return {
            "prompt_id": rec["prompt_id"],
            "domain": rec["domain"],
            "split": rec["split"],
            "prompt": prompt,
            "completion": completion,
            "n_tokens": n_tokens,
        }

    ds = ds.map(convert, num_proc=8, remove_columns=ds.column_names)

    lengths = sorted(ds["n_tokens"])
    n = len(lengths)
    print(f"token lengths over {n} rows: "
          f"p50={lengths[n // 2]} p90={lengths[int(n * .9)]} "
          f"p99={lengths[int(n * .99)]} max={lengths[-1]}")

    kept = ds.filter(lambda r: r["n_tokens"] <= args.max_length, num_proc=8)
    print(f"dropped {n - len(kept)}/{n} rows over {args.max_length} tokens")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for split in ("train", "eval"):
        part = kept.filter(lambda r: r["split"] == split)
        path = out_dir / f"{split}.jsonl"
        part.to_json(path, lines=True)
        print(f"{path}: {len(part)} rows")

    sample = kept[0]
    print("\n--- sample (truncated) ---")
    print(json.dumps(sample["prompt"][-200:]))
    print(json.dumps(sample["completion"][:200]))


if __name__ == "__main__":
    main()
