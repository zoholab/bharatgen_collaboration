import os
import sys
import json
import argparse
import pickle
from typing import List, Tuple
from tqdm import tqdm
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samd.wordgroup_sam import WordGroupAwareSAM


def load_processed_hindi_data(input_file: str) -> List[dict]:
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    for i, entry in enumerate(data[:5]):
        if 'tokens' not in entry or 'word_group_boundaries' not in entry:
            print(f"Warning: Entry {i} missing required keys")
            print(f"Keys found: {entry.keys()}")
    
    return data

TARGET_STATES_PER_SHARD = 2_000_000


def dump_sam(shard_dir: str, sam: WordGroupAwareSAM):

    os.makedirs(shard_dir, exist_ok=True)

    num_states = len(sam.states)
    num_shards = math.ceil(num_states / TARGET_STATES_PER_SHARD)

    print(f"SAVING SAM:")
    print(f"  total states = {num_states}")
    print(f"  shard size   = {TARGET_STATES_PER_SHARD}")
    print(f"  num shards   = {num_shards}")

    if not isinstance(sam.input_ids, torch.Tensor):
        sam.input_ids = torch.tensor(sam.input_ids, dtype=torch.int32)

    if not isinstance(sam.word_boundaries, torch.Tensor):
        sam.word_boundaries = torch.tensor(sam.word_boundaries, dtype=torch.bool)

    meta = {
        "n_predicts": sam.n_predicts,
        "num_states": num_states,
        "num_shards": num_shards,
        "last": sam.last,
        "max_length": sam.max_length,
        "input_ids": sam.input_ids,
        "word_boundaries": sam.word_boundaries,
        "shard_size": TARGET_STATES_PER_SHARD,
    }

    torch.save(meta, os.path.join(shard_dir, "sam_meta.pt"))

    for shard_id in range(num_shards):

        start = shard_id * TARGET_STATES_PER_SHARD
        end = min(num_states, (shard_id + 1) * TARGET_STATES_PER_SHARD)

        links = []
        lengths = []
        min_endpos = []
        transitions = []

        for st in sam.states[start:end]:
            links.append(st.link)
            lengths.append(st.length)
            min_endpos.append(st.min_endpos)
            transitions.append(st.next)

        shard_obj = {
            "links": torch.tensor(links, dtype=torch.int32),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "min_endpos": torch.tensor(min_endpos, dtype=torch.int32),
            "transitions": transitions,
        }

        shard_path = os.path.join(shard_dir, f"sam_states_{shard_id:04d}.pt")
        torch.save(shard_obj, shard_path)

        print(f"saved shard {shard_id+1}/{num_shards} ({start}–{end})")




def main():
    parser = argparse.ArgumentParser(
        description="Build word-group-aware SAM from processed Hindi data"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default=' ',
        help='Path to processed Hindi data (JSON format)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default= ' ',
        help='Path to save the SAM pickle file'
    )
    parser.add_argument(
        '--n_predicts', 
        type=int, 
        default=15,
        help='Maximum number of tokens to predict (will stop at word boundaries)'
    )
    parser.add_argument(
        '--eos_token', 
        type=int, 
        default=2,
        help='End of sequence token ID'
    )
    
    args = parser.parse_args()
    
    hindi_data = load_processed_hindi_data(args.input_file)
    
    print(f"\nBuilding word-group-aware SAM with n_predicts={args.n_predicts}...")
    print(f"Total sequences: {len(hindi_data)}")
    
    sam = WordGroupAwareSAM.build(
        batch_data=hindi_data,
        eos_token=args.eos_token,
        n_predicts=args.n_predicts,
        verbose=True
    )
    
    print(f"\nSAM Statistics:")
    print(f"Total states: {len(sam.states)}")
    print(f"Total tokens: {len(sam.input_ids)}")
    print(f"Word boundaries: {sum(sam.word_boundaries)}")
    print(f"Max length: {sam.max_length}")
    
    dump_sam(args.output_path, sam)
    
    print(f"Successfully created word-group-aware SAM!")
    print(f"Saved to: {args.output_path}")


if __name__ == '__main__':
    main()
