import json
import argparse
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

def parse_word_groups(text: str) -> List[Tuple[str, bool]]:
    """
    Parse text and identify word groups.
    Returns list of (word, is_group_end) tuples.
    
    Example: "कर्नाटक##ने भारतीय शास्त्रीय संगीत##के"
    Should return: [("कर्नाटक", False), ("ने", True), ("भारतीय", False), ("शास्त्रीय", False), ("संगीत", False), ("के", True)]
    """
    # First, split by spaces to get word segments
    segments = text.split()
    result = []
    
    for segment in segments:
        if "##" in segment:
            # This segment contains multiple words in a group
            words_in_segment = segment.split("##")
            for i, word in enumerate(words_in_segment):
                if word.strip():  # Skip empty parts
                    # All words except the last one are not group ends
                    is_group_end = (i == len(words_in_segment) - 1)
                    result.append((word.strip(), is_group_end))
        else:
            # Single word, it's a complete group by itself
            result.append((segment.strip(), True))
    
    return result

def create_word_boundaries_from_groups(tokens: List[int], tokenizer, word_groups: List[Tuple[str, bool]], cleaned_text: str) -> List[bool]:
    """
    Create boundary markers for word groups in tokenized sequence.
    Uses precise character-offset alignment when available, falls back to word-based alignment.
    """
    boundaries = [False] * len(tokens)
    
    if not word_groups or not tokens:
        return boundaries
    
    try:
        # Try to use offset mapping for precise alignment
        full_encoding = tokenizer(cleaned_text, return_offsets_mapping=True, add_special_tokens=True)
        
        if 'offset_mapping' in full_encoding and len(full_encoding['input_ids']) == len(tokens):
            return create_boundaries_with_offsets(tokens, tokenizer, word_groups, cleaned_text, full_encoding['offset_mapping'])
        else:
            return create_boundaries_word_based(tokens, tokenizer, word_groups, cleaned_text)
            
    except Exception as e:
        print(f"Warning: Using fallback boundary alignment due to: {e}")
        return create_boundaries_word_based(tokens, tokenizer, word_groups, cleaned_text)

def create_boundaries_with_offsets(tokens: List[int], tokenizer, word_groups: List[Tuple[str, bool]], 
                                 cleaned_text: str, offset_mapping: List[Tuple[int, int]]) -> List[bool]:
    """Create boundaries using character offset mapping (most accurate)."""
    
    boundaries = [False] * len(tokens)
    
    # Find character positions where each word ends in the cleaned text
    word_end_positions = []
    char_pos = 0
    
    for word, is_group_end in word_groups:
        word = word.strip()
        
        # Find this word in the cleaned text starting from current position
        word_start = cleaned_text.find(word, char_pos)
        if word_start != -1:
            word_end = word_start + len(word)
            word_end_positions.append((word_end - 1, is_group_end))  # -1 to get last char of word
            char_pos = word_end
            
            # Skip spaces to next word
            while char_pos < len(cleaned_text) and cleaned_text[char_pos] == ' ':
                char_pos += 1
        else:
            print(f"Warning: Could not find word '{word}' in cleaned text")
    
    # Mark boundaries based on which tokens contain the end of word groups
    for i, (start, end) in enumerate(offset_mapping):
        if i >= len(boundaries):
            break
            
        # Skip special tokens (they typically have (0,0) offsets)
        if start == 0 and end == 0 and i > 0:
            continue
        
        # Check if this token contains the end of any word group
        for word_end_pos, is_group_end in word_end_positions:
            if is_group_end and start <= word_end_pos < end:
                boundaries[i] = True
                break
    
    # Ensure last meaningful token is marked as boundary
    last_meaningful_idx = len(boundaries) - 1
    while (last_meaningful_idx >= 0 and 
           last_meaningful_idx < len(tokens) and
           tokens[last_meaningful_idx] in [tokenizer.eos_token_id, tokenizer.pad_token_id]):
        last_meaningful_idx -= 1
    
    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True
    
    return boundaries

def create_boundaries_word_based(tokens: List[int], tokenizer, word_groups: List[Tuple[str, bool]], 
                                cleaned_text: str) -> List[bool]:
    """Create boundaries using improved word-based alignment (fallback method)."""
    
    boundaries = [False] * len(tokens)
    
    if not word_groups:
        return boundaries
    
    # Build word-to-group mapping
    words_with_boundaries = []
    for word, is_group_end in word_groups:
        words_with_boundaries.append((word.strip(), is_group_end))
    
    # Try to align with tokens by decoding individual tokens
    try:
        # Decode each token to understand the text structure
        token_texts = []
        for i, token_id in enumerate(tokens):
            if hasattr(tokenizer, 'bos_token_id') and token_id == tokenizer.bos_token_id:
                token_texts.append("[BOS]")
            elif hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
                token_texts.append("[EOS]")
            elif hasattr(tokenizer, 'pad_token_id') and token_id == tokenizer.pad_token_id:
                token_texts.append("[PAD]")
            else:
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    token_texts.append(token_text)
                except:
                    token_texts.append("[UNK]")
        
        # Reconstruct text and find word boundaries
        reconstructed_text = ""
        token_to_char_map = []
        
        for i, token_text in enumerate(token_texts):
            if token_text.startswith("[") and token_text.endswith("]"):
                # Special token
                token_to_char_map.append((len(reconstructed_text), len(reconstructed_text)))
            else:
                start_pos = len(reconstructed_text)
                reconstructed_text += token_text
                end_pos = len(reconstructed_text)
                token_to_char_map.append((start_pos, end_pos))
        
        # Find where each word ends in the reconstructed text
        char_pos = 0
        for word, is_group_end in words_with_boundaries:
            # Find this word in reconstructed text
            word_start = reconstructed_text.find(word, char_pos)
            if word_start != -1:
                word_end = word_start + len(word) - 1  # Last character of word
                
                if is_group_end:
                    # Find which token contains this character position
                    for i, (token_start, token_end) in enumerate(token_to_char_map):
                        if token_start <= word_end < token_end:
                            boundaries[i] = True
                            break
                
                char_pos = word_start + len(word)
                # Skip spaces
                while char_pos < len(reconstructed_text) and reconstructed_text[char_pos] == ' ':
                    char_pos += 1
        
    except Exception as e:
        print(f"Warning: Word-based alignment failed ({e}), using simple distribution")
        # Simple fallback: distribute boundaries evenly
        group_count = sum(1 for _, is_end in word_groups if is_end)
        if group_count > 0:
            tokens_per_group = len(tokens) / group_count
            for i in range(group_count):
                boundary_pos = min(int((i + 1) * tokens_per_group) - 1, len(tokens) - 1)
                if boundary_pos >= 0:
                    boundaries[boundary_pos] = True
    
    # Always mark the last meaningful token as boundary
    last_meaningful_idx = len(boundaries) - 1
    while (last_meaningful_idx >= 0 and 
           last_meaningful_idx < len(tokens) and
           hasattr(tokenizer, 'eos_token_id') and hasattr(tokenizer, 'pad_token_id') and
           tokens[last_meaningful_idx] in [tokenizer.eos_token_id, tokenizer.pad_token_id]):
        last_meaningful_idx -= 1
    
    if last_meaningful_idx >= 0:
        boundaries[last_meaningful_idx] = True
    
    return boundaries

def process_hindi_data(input_file: str, output_file: str, model_name: str = "/workspace/pranav-shinde/download/Airavata"):
    """
    Process Hindi JSONL data and create word-group-aware tokenized data.
    Replace ## with whitespace and mark word boundaries based on original groups.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                original_hindi_text = data['translation']['hin']
                
                # Parse word groups from original text
                word_groups = parse_word_groups(original_hindi_text)
                
                # Create cleaned text by replacing ## with whitespace
                cleaned_text = original_hindi_text.replace('##', ' ')
                
                # Tokenize the cleaned text (without ##)
                tokens = tokenizer(cleaned_text, return_tensors=None)['input_ids']
                
                # Create word boundaries based on original word groups and tokenized sequence
                boundaries = create_word_boundaries_from_groups(tokens, tokenizer, word_groups, cleaned_text)
                
                processed_entry = {
                    'original_text': original_hindi_text,  # Keep original with ##
                    'cleaned_text': cleaned_text,          # Text without ##
                    'tokens': tokens,
                    'word_group_boundaries': boundaries,
                    'word_groups': word_groups,            # Store parsed word groups
                    'original_data': data
                }
                
                processed_data.append(processed_entry)
                
                if (line_num + 1) % 1000 == 0:
                    print(f"Processed {line_num + 1} lines...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(processed_data)} entries and saved to {output_file}")
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Hindi data for word-group-aware SAM")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default="/nfs/kundeshwar/pranav-shinde/download/Airavata", help="Tokenizer model name")
    
    args = parser.parse_args()
    
    process_hindi_data(args.input, args.output, args.model)
