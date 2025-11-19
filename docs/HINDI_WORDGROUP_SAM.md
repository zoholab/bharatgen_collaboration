# Word-Group-Aware SAM for Hindi

This directory contains scripts for building and using a word-group-aware Suffix Automaton (SAM) for Hindi text generation. The key feature is that draft predictions from SAM **respect word group boundaries** as marked in the training data.

## Overview

### What are Word Groups?

In the Hindi dataset (`processed_nios.jsonl`), words are organized into groups using the `##` delimiter in the original text. For example:

```
कर्नाटक##ने भारतीय##शास्त्रीय##संगीत##के
```

Here:
- `कर्नाटक##ने` is one word group (2 words)
- `भारतीय##शास्त्रीय##संगीत##के` is another word group (4 words)
- A single word without `##` is also a word group

### Why Word Group Boundaries?

The word-group-aware SAM ensures that during speculative decoding:
1. Draft predictions **stop at word group boundaries**
2. Each prediction step generates tokens up to the next boundary
3. This provides more meaningful and coherent drafts for Hindi text

## Data Format

The `processed_nios.jsonl` file contains entries with:

```json
{
  "original_text": "हिंदुस्तानी शास्त्रीय संगीत भारतीय उपमहाद्वीप##के...",
  "cleaned_text": "हिंदुस्तानी शास्त्रीय संगीत भारतीय उपमहाद्वीप के...",
  "tokens": [1, 44131, 37325, ...],
  "word_group_boundaries": [false, false, false, true, ...]
}
```

- `tokens`: Tokenized representation
- `word_group_boundaries`: Boolean array marking word group endings
  - `true` = this token marks the end of a word group
  - `false` = this token is in the middle of a word group

## Files

### Scripts

1. **`tools/gen_sam_hindi_wordgroup.py`**
   - Builds word-group-aware SAM from processed data
   - Stores boundary information alongside tokens
   - Modified `gen_draft()` to stop at boundaries

2. **`tests/test_samd_hindi_wordgroup.py`**
   - Inference script using word-group-aware SAM
   - Custom `WordGroupAwareDraftModel` respects boundaries
   - Provides detailed performance metrics

### Shell Scripts

3. **`scripts/build_hindi_sam.sh`**
   - Convenient wrapper to build SAM
   - Configure paths and parameters

4. **`scripts/test_hindi_sam.sh`**
   - Convenient wrapper to run inference
   - Configure model and SAM paths

## Usage

### Step 1: Build Word-Group-Aware SAM

```bash
cd /nfs/kundeshwar/pranav-shinde/SAM-Decoding

# Using the shell script (recommended)
./scripts/build_hindi_sam.sh

# Or directly with Python
python tools/gen_sam_hindi_wordgroup.py \
    --input_file /nfs/kundeshwar/pranav-shinde/SAM-Decoding/downloads/processed_nios.jsonl \
    --output_path downloads/sam_hindi_wordgroup.pkl \
    --n_predicts 15 \
    --eos_token 2
```

**Parameters:**
- `--input_file`: Path to processed_nios.jsonl
- `--output_path`: Where to save the SAM pickle file
- `--n_predicts`: Max tokens to predict (will stop at boundaries)
- `--eos_token`: EOS token ID (default: 2 for Airavata model)

### Step 2: Run Inference

```bash
# Using the shell script (recommended)
./scripts/test_hindi_sam.sh

# Or directly with Python
python tests/test_samd_hindi_wordgroup.py \
    --model_path /nfs/kundeshwar/pranav-shinde/download/Airavata \
    --sam_path downloads/sam_hindi_wordgroup.pkl \
    --samd_n_predicts 15 \
    --max_new_tokens 256 \
    --tree_method eagle2 \
    --dtype float16 \
    --device cuda \
    --prompt "हिंदुस्तानी शास्त्रीय संगीत के बारें में बताओ।"
```

**Parameters:**
- `--model_path`: Path to the Hindi language model (Airavata)
- `--sam_path`: Path to word-group-aware SAM file
- `--samd_n_predicts`: Max draft tokens (stops at boundaries)
- `--max_new_tokens`: Total tokens to generate
- `--tree_method`: Fallback method (token_recycle or eagle2)
- `--prompt`: Hindi text prompt

## How It Works

### Building SAM

The `WordGroupAwareSAM` class extends `StaticSAM`:

1. **Stores boundary info**: Maintains `word_boundaries` list alongside tokens
2. **Processes batch data**: Reads tokens and boundaries from JSON
3. **Builds suffix automaton**: Standard SAM construction with boundaries tracked

### Draft Generation

During inference, `gen_draft()` method:

1. **Finds longest match**: Uses suffix automaton to find matching prefix
2. **Generates tokens**: Predicts next tokens from SAM
3. **Stops at boundary**: Checks `word_boundaries[position]` and stops when `true`
4. **Returns draft**: Provides tokens up to next word group boundary

## Code Architecture

### Key Classes

1. **`WordGroupAwareSAM`** (in `gen_sam_hindi_wordgroup.py`)
   ```python
   class WordGroupAwareSAM(StaticSAM):
       def __init__(self, n_predicts: int = 40)
       def add_tokens_with_boundaries(tokens, boundaries)
       def gen_draft(index, start_token)  # Stops at boundaries!
   ```

2. **`WordGroupAwareDraftModel`** (in `test_samd_hindi_wordgroup.py`)
   ```python
   class WordGroupAwareDraftModel(DraftModel):
       def lookup(start_token)  # Uses word-aware gen_draft
   ```
