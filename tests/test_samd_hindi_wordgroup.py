"""
Inference script for word-group-aware SAM with Hindi text.
This ensures predictions stop at word group boundaries.
"""
import os
import sys
import argparse
import pickle
import torch
import time
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel
from samd.wordgroup_sam import WordGroupAwareSAM
from samd.sam import DynSAM


class WordGroupAwareDraftModel(DraftModel):
    """
    Draft model that respects word group boundaries during inference.
    """
    
    def __init__(self, config, sam_static=None, lm=None, dtype=None, device=None, tokenizer=None):
        """Initialize with original dynamic SAM."""
        # Create original dynamic SAM (no word-group awareness needed here)
        sam_dyn = DynSAM(config.n_predicts)
        
        # Store tokenizer for decoding
        self.tokenizer = tokenizer
        
        # Initialize statistics
        self.static_sam_accepts = 0
        self.dynamic_sam_accepts = 0
        self.eagle_accepts = 0
        
        # Call parent init
        super().__init__(
            config=config,
            sam_dyn=sam_dyn,
            sam_static=sam_static,
            lm=lm,
            dtype=dtype,
            device=device
        )
    
    def update(self, tokens=None, last_hidden_states=None, tree_tokens=None, tree_logits=None):
        """Update SAMs with accepted tokens."""
        if tokens is not None:
            tokens_list = tokens.tolist()
            if self.tokenizer:
                # Decode each token individually for better visibility
                decoded_tokens = [self.tokenizer.decode([t], skip_special_tokens=True) for t in tokens_list]
                decoded_text = self.tokenizer.decode(tokens_list, skip_special_tokens=True)
                print(f"\n  📝 Updating with {len(tokens_list)} accepted tokens:")
                print(f"     Tokens: {tokens_list}")
                print(f"     Decoded: {decoded_tokens}")
                print(f"     Combined: '{decoded_text}'")
            else:
                print(f"\n  📝 Updating Dynamic SAM with {len(tokens_list)} accepted tokens: {tokens_list}")
            self.sam_dyn.add_tokens(tokens_list)
            self.sam_static.transfer_tokens(tokens_list)
        
        self.tree_model.update(
            tokens=tokens,
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens,
            tree_logits=tree_logits,
        )
    
    def lookup(self, start_token: int):
        """
        Lookup draft tokens, respecting word boundaries.
        """
        print(f"\n  🔍 Lookup for token {start_token}")
        
        # Decode start token if tokenizer available
        if self.tokenizer:
            start_text = self.tokenizer.decode([start_token], skip_special_tokens=True)
            print(f"    📝 Start token: {start_token} → '{start_text}'")
        
        # Check dynamic SAM first
        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        print(f"    🔄 Dynamic SAM: index={index_dyn}, match_length={match_dyn}")
        
        # Check static SAM
        index_static, match_static = self.sam_static.lookup(start_token)
        match_static -= self.len_bias
        print(f"    📚 Static SAM: index={index_static}, match_length={match_static} (after bias)")
        
        # Decide which SAM to use
        if max(match_dyn, match_static) >= self.len_threshold:
            from samd.draft import CandidateType
            
            if match_dyn >= match_static:
                print(f"    ✅ Using DYNAMIC SAM (match_dyn={match_dyn} >= match_static={match_static})")
                self.dynamic_sam_accepts += 1
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
            else:
                print(f"    ✅ Using STATIC SAM (match_static={match_static} > match_dyn={match_dyn})")
                self.static_sam_accepts += 1
                # Use word-group-aware gen_draft from static SAM
                seq = self.sam_static.gen_draft(index_static, start_token)
            
            # Decode and print the draft with individual tokens
            if self.tokenizer:
                # Filter out padding tokens (0) for better display
                non_zero_seq = [t for t in seq if t != 0]
                draft_tokens_decoded = [self.tokenizer.decode([t], skip_special_tokens=True) for t in non_zero_seq]
                draft_text = self.tokenizer.decode(non_zero_seq, skip_special_tokens=True)
                print(f"    📄 Draft: {len(non_zero_seq)} tokens")
                print(f"       Token IDs: {non_zero_seq}")
                print(f"       Decoded tokens: {draft_tokens_decoded}")
                print(f"       Combined text: '{draft_text}'")
            
            return (CandidateType.sequence, seq, {})
        else:
            print(f"    ⚠️  Neither SAM meets threshold (best={max(match_dyn, match_static)} < {self.len_threshold}), using TREE/EAGLE model")
            self.eagle_accepts += 1
            # Fall back to tree model
            from samd.draft import CandidateType
            return (CandidateType.tree,) + self.tree_model.gen_draft(start_token)


def load_wordgroup_sam(path: str):
    """Load word-group-aware SAM from pickle file."""
    print(f"Loading word-group-aware SAM from {path}...")
    start = time.perf_counter()
    
    with open(path, "rb") as f:
        sam = pickle.load(f)
    
    end = time.perf_counter()
    print(f"Loaded SAM in {end - start:.2f} seconds")
    print(f"  States: {len(sam.states)}")
    print(f"  Tokens: {len(sam.input_ids)}")
    print(f"  Word boundaries: {sum(sam.word_boundaries)}")
    
    return sam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with word-group-aware SAM for Hindi"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="/nfs/kundeshwar/pranav-shinde/download/Airavata",
        help='Path to the language model'
    )
    parser.add_argument(
        '--sam_path', 
        type=str, 
        default="downloads/sam_hindi_wordgroup.pkl",
        help='Path to word-group-aware SAM pickle file'
    )
    parser.add_argument(
        '--samd_n_predicts', 
        type=int, 
        default=15,
        help='Number of tokens to predict (will stop at word boundaries)'
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=512,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--max_cache_len', 
        type=int, 
        default=2048,
        help='Maximum cache length'
    )
    parser.add_argument(
        "--tree_method", 
        type=str, 
        default="eagle2",
        choices=["token_recycle", "eagle2"],
        help='Tree method for fallback'
    )
    parser.add_argument(
        "--tree_model_path", 
        type=str, 
        default="/nfs/kundeshwar/pranav-shinde/SAM-Decoding/downloads/airavata_bs1/state_20",
        help='Path to tree model (if using eagle2)'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='float16', 
        choices=['float16', 'float32'],
        help='Model dtype'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda", 
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Hindi prompt for testing'
    )
    
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    
    return args


@torch.inference_mode()
def samd_generate_wordgroup(args, inputs, model, tokenizer, sam):
    """
    Generate text using word-group-aware SAM.
    """
    from samd.sam import DynSAM
    
    # Create configuration
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
    )
    
    # Create word-group-aware draft model
    draft = WordGroupAwareDraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device,
        tokenizer=tokenizer
    )
    
    # Create SAMD model
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
    )
    samd_model.eval()
    
    # Generation configuration
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0
    )
    
    print("\n" + "="*80)
    print("Starting generation with word-group-aware SAM...")
    print("="*80)
    
    st = time.perf_counter()
    outputs = samd_model.generate(**inputs, generation_config=gen_config)
    ed = time.perf_counter()
    
    # Decode response
    response = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
    
    # Print results
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"Generation time: {ed - st:.2f} seconds")
    print(f"Decode steps: {outputs.decode_steps}")
    print(f"Decode tokens: {outputs.decode_tokens}")
    print(f"Tokens per second: {outputs.decode_tokens / (ed - st):.2f}")
    print(f"Average accept length per step: {sum(outputs.accepet_length_per_step) / len(outputs.accepet_length_per_step):.2f}")
    print(f"\nAccept lengths per step: {outputs.accepet_length_per_step}")
    
    # Print acceptance statistics
    print("\n" + "="*80)
    print("ACCEPTANCE STATISTICS")
    print("="*80)
    total_lookups = draft.static_sam_accepts + draft.dynamic_sam_accepts + draft.eagle_accepts
    print(f"📊 Static SAM accepted:  {draft.static_sam_accepts:3d} times ({draft.static_sam_accepts/total_lookups*100:5.1f}%)")
    print(f"📊 Dynamic SAM accepted: {draft.dynamic_sam_accepts:3d} times ({draft.dynamic_sam_accepts/total_lookups*100:5.1f}%)")
    print(f"📊 EAGLE accepted:       {draft.eagle_accepts:3d} times ({draft.eagle_accepts/total_lookups*100:5.1f}%)")
    print(f"📊 Total lookups:        {total_lookups:3d}")
    print("="*80)
    
    print("\n" + "-"*80)
    print("GENERATED TEXT")
    print("-"*80)
    print(response)
    print("-"*80 + "\n")
    
    return outputs


@torch.inference_mode()
def baseline_generate(args, inputs, model, tokenizer):
    """
    Baseline generation without SAM for comparison.
    """
    model.eval()
    
    from samd import SamdGenerationConfig
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0
    )
    
    print("\n" + "="*80)
    print("Baseline generation (without SAM)...")
    print("="*80)
    
    st = time.perf_counter()
    tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)[0]
    ed = time.perf_counter()
    
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    
    print("\n" + "-"*80)
    print("BASELINE RESULTS")
    print("-"*80)
    print(f"Generation time: {ed - st:.2f} seconds")
    print(f"Tokens generated: {len(tokens) - inputs.input_ids.shape[-1]}")
    print(f"\nGenerated text:")
    print(response)
    print("-"*80 + "\n")
    
    return tokens


def main():
    args = parse_args()
    
    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Load word-group-aware SAM
    if args.sam_path and os.path.exists(args.sam_path):
        sam = load_wordgroup_sam(args.sam_path)
    else:
        print(f"Warning: SAM file not found at {args.sam_path}")
        print("Proceeding without static SAM (will use dynamic SAM only)")
        sam = None
    
    # Prepare prompt
    if args.prompt:
        prompt = args.prompt
    else:
        # Default Hindi test prompt
        prompt = "हिंदुस्तानी शास्त्रीय संगीत"
    
    print(f"\n{'='*80}")
    print(f"PROMPT")
    print(f"{'='*80}")
    print(prompt)
    print(f"{'='*80}\n")
    
    # Tokenize input
    inputs = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt"
    ).to(args.device)
    
    print(f"Input tokens: {inputs.input_ids.shape[-1]}")
    
    # Run baseline generation for comparison
    # baseline_generate(args, inputs, model, tokenizer)
    
    # Run word-group-aware SAM generation
    if sam is not None:
        samd_generate_wordgroup(args, inputs, model, tokenizer, sam)
    else:
        print("Skipping SAM generation (no SAM loaded)")


if __name__ == '__main__':
    main()
