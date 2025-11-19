"""
Word-group-aware SAM class for Hindi text generation.
This module can be imported by both build and inference scripts.
"""
from typing import List
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import samd modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samd.sam.static_sam import StaticSAM


class WordGroupAwareSAM(StaticSAM):
    """
    SAM that respects word group boundaries during draft generation.
    
    This class extends StaticSAM to store word group boundary information
    and modify draft generation to stop at linguistic boundaries.
    """
    
    def __init__(self, n_predicts: int = 40):
        super().__init__(n_predicts)
        # Store word group boundary information
        # Maps position in input_ids to whether it's a boundary
        self.word_boundaries: List[bool] = [False]  # Start with False for initial state
        
    def add_batch_tokens_with_boundaries(
        self, 
        batch_data: List[dict], 
        eos_token: int, 
        verbose: bool = True
    ):
        """
        Add tokens with their word group boundary information.
        
        Args:
            batch_data: List of dicts with 'tokens' and 'word_group_boundaries' keys
            eos_token: End of sequence token
            verbose: Show progress bar
        """
        for data in tqdm(batch_data, desc="Building word-group-aware SAM...", disable=not verbose):
            tokens = data['tokens']
            boundaries = data['word_group_boundaries']
            
            # Validate that tokens and boundaries have same length
            if len(tokens) != len(boundaries):
                print(f"Warning: Token length {len(tokens)} != boundary length {len(boundaries)}, skipping")
                continue
                
            self.add_tokens_with_boundaries(tokens, boundaries)
            
            # Add EOS token if not present
            if tokens[-1] != eos_token:
                self.add_tokens_with_boundaries([eos_token], [True])  # EOS is always a boundary
    
    def add_tokens_with_boundaries(self, tokens: List[int], boundaries: List[bool]):
        """
        Add tokens and their corresponding word group boundaries.
        """
        for token, is_boundary in zip(tokens, boundaries):
            self.transfer_cur_state(token)
            self.add_state(token)
            self.word_boundaries.append(is_boundary)
        self.input_ids.extend(tokens)
    
    def gen_draft(self, index: int, start_token: int):
        """
        Generate draft tokens, stopping at the next word group boundary.
        
        Returns:
            List of predicted token IDs, stopping at next word boundary
        """
        print(f"    📚 [STATIC SAM] Generating draft...")
        
        if index == 0:
            # No match found, return empty draft
            print(f"    📚 [STATIC SAM] No match found, returning empty draft")
            return [start_token] + [0] * (self.n_predicts - 1)
        
        endpos = self.states[index].min_endpos
        
        # Start from position after the match
        start_pos = endpos + 1
        pred_ids = [start_token]
        
        # Generate tokens until we hit a word boundary or reach n_predicts
        tokens_generated = 0
        current_pos = start_pos
        
        while tokens_generated < self.n_predicts - 1 and current_pos < len(self.input_ids):
            token = self.input_ids[current_pos]
            pred_ids.append(token)
            tokens_generated += 1
            
            # Check if this position is a word boundary
            if current_pos < len(self.word_boundaries) and self.word_boundaries[current_pos]:
                # Stop at word group boundary
                print(f"    📚 [STATIC SAM] Stopped at boundary after {len(pred_ids)} tokens")
                break
            
            current_pos += 1
        
        # Pad with zeros if we haven't reached n_predicts
        while len(pred_ids) < self.n_predicts:
            pred_ids.append(0)
        
        print(f"    📚 [STATIC SAM] Draft: {len([t for t in pred_ids if t != 0])} tokens (first 5: {pred_ids[:5]})")
        return pred_ids
    
    @staticmethod
    def build(
        batch_data: List[dict],
        eos_token: int,
        n_predicts: int = 40,
        verbose: bool = True
    ):
        """
        Build word-group-aware SAM from batch data.
        
        Args:
            batch_data: List of dicts with 'tokens' and 'word_group_boundaries' keys
            eos_token: End of sequence token
            n_predicts: Maximum number of tokens to predict
            verbose: Show progress
            
        Returns:
            WordGroupAwareSAM instance
        """
        sam = WordGroupAwareSAM(n_predicts)
        sam.add_batch_tokens_with_boundaries(batch_data, eos_token, verbose)
        return sam
