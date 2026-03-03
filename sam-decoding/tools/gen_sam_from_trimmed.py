"""
Build SAM from cleaned text in trimmed.jsonl (JSON array format).
Uses all cleaned text entries from the file; does not modify gen_sam_alpaca.py.
"""
import json
import argparse
from transformers import AutoTokenizer
from samd import build_sam, dump_sam

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='models/main')
parser.add_argument('--trimmed_path', type=str, default='downloads/trimmed.jsonl')
parser.add_argument('--cutoff_len', type=int, default=2048)
parser.add_argument('--sam_path', type=str, default="downloads/new_sam.pkl")
args = parser.parse_args()


def load_cleaned_texts(path: str):
    """Load all cleaned text entries from trimmed JSON file (array format)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data:
        if isinstance(item, dict) and "cleaned_text" in item:
            text = item["cleaned_text"]
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


# Load all cleaned text from trimmed file
cleaned_texts = load_cleaned_texts(args.trimmed_path)
print(f"Loaded {len(cleaned_texts)} cleaned text entries")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


def tokenize_fn(text, add_eos_token=False):
    result = tokenizer(
        text,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    return result


batch_tokens = [tokenize_fn(text)["input_ids"] for text in cleaned_texts]
for i in range(len(tokenizer)):
    batch_tokens.append([i])

sam = build_sam(batch_tokens, tokenizer.eos_token_id)
dump_sam(args.sam_path, sam)
print(f"Saved SAM to {args.sam_path}")
