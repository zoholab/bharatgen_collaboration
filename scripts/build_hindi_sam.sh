#!/bin/bash

# Build word-group-aware SAM for Hindi data
# This script builds a SAM that respects word group boundaries

# Default paths - modify as needed
INPUT_FILE="/nfs/kundeshwar/pranav-shinde/SAM-Decoding/downloads/processed_file.jsonl"
OUTPUT_PATH="/nfs/kundeshwar/pranav-shinde/SAM-Decoding/downloads/sam_hindi_wordgroup_wiki.pkl"
N_PREDICTS=15
# EOS TOKEN for Airavata tokenizer
EOS_TOKEN=2

echo "Building word-group-aware SAM for Hindi..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_PATH"
echo "n_predicts: $N_PREDICTS"
echo ""

python tools/gen_sam_hindi_wordgroup.py \
    --input_file "$INPUT_FILE" \
    --output_path "$OUTPUT_PATH" \
    --n_predicts "$N_PREDICTS" \
    --eos_token "$EOS_TOKEN"

echo ""
echo "Done! SAM saved to $OUTPUT_PATH"
