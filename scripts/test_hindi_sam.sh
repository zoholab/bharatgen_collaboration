#!/bin/bash

# Test word-group-aware SAM inference for Hindi
# This script runs inference that respects word group boundaries

# Default paths - modify as needed
MODEL_PATH="/nfs/kundeshwar/pranav-shinde/download/Airavata"
SAM_PATH="downloads/sam_hindi_wordgroup.pkl"
N_PREDICTS=15
MAX_NEW_TOKENS=256
DRAFT_PATH="/nfs/kundeshwar/pranav-shinde/SAM-Decoding/downloads/airavata_bs1/state_20"


# Default Hindi prompt
PROMPT="हिंदुस्तानी शास्त्रीय संगीत के बारें में बताओ।"

echo "Testing word-group-aware SAM inference..."
echo "Model: $MODEL_PATH"
echo "SAM: $SAM_PATH"
echo "Prompt: $PROMPT"
echo ""

CUDA_VISIBLE_DEVICES=3 python tests/test_samd_hindi_wordgroup.py \
    --model_path "$MODEL_PATH" \
    --sam_path "$SAM_PATH" \
    --samd_n_predicts "$N_PREDICTS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --tree_method "eagle2" \
    --tree_model_path "$DRAFT_PATH" \
    --dtype "float16" \
    --device "cuda" \
    --prompt "$PROMPT"

echo ""
echo "Inference complete!"
