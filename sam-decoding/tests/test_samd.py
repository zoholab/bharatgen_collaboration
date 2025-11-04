import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    GenerationMixin,
    LlamaConfig,
    LlamaTokenizer
)
from samd import (
    SamdConfig, 
    SamdModel, 
    SamdGenerationConfig,
    DraftModel,
    load_sam
)
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sam_path', type=str, default=None)
    parser.add_argument('--samd_n_predicts', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_cache_len', type=int, default=2048)
    parser.add_argument("--tree_method", type=str, default="eagle2")
    parser.add_argument("--tree_model_path", type=str, default=None)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    return args

@torch.inference_mode()
def generate(args, inputs, model, tokenizer):
    model.eval()
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0
    )
    st = time.perf_counter()
    tokens = model.generate(**inputs, generation_config=gen_config)[0]
    ed = time.perf_counter()
    response = tokenizer.decode(tokens)
    print("model inference time use: {} seconds".format(ed - st))
    print("model response:\n{}".format(repr(response)))


@torch.inference_mode()
def samd_generate(args, inputs, model, tokenizer):
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    sam = load_sam(args.sam_path) if args.sam_path is not None else None


    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
    )

    draft = DraftModel(
        samd_config, 
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device
    )

    samd_model = SamdModel(
        samd_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
    )
    samd_model.eval()
    
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
    )
    st = time.perf_counter()

    gen = samd_model.stream_generate(**inputs, generation_config=gen_config)
    for chunk in gen:
        print(chunk["ids"], end="", flush=True)
    ed = time.perf_counter()
    # response = tokenizer.decode(outputs.output_ids[0])
    # print("model inference time use: {} seconds".format(ed - st))
    # print("samd_model response:\n{}".format(repr(response)))
    # print("decode_steps: {}".format(outputs.decode_steps))
    # print("decode_tokens: {}".format(outputs.decode_tokens))
    # print("accepect_length_per_step: {}".format(outputs.accepet_length_per_step))

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=args.dtype, 
        device_map=args.device,
    )
    model.eval()
    
    # prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Give three tips for staying healthy.\n\nASSISTANT: "]
    
    # prompts = ['A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\nUSER: Please generate the following: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10".\n\nASSISTANT: ']
    
    prompts = ["write a sentence about india"]

    inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt"
    ).to(args.device)
    # print(dict(sorted(tokenizer.get_vocab().items(), key=lambda x : x[-1])))
    # exit()
    # for i in range(0,50):
    #     print(f"The {i}th token is",tokenizer.decode([i]))
    # exit()

    # generate(args, inputs, model, tokenizer)
    samd_generate(args, inputs, model, tokenizer)

if __name__ == '__main__':
    main()
