from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch 
import argparse 

parser=argparse.ArgumentParser()
parser.add_argument('--base_model_path',type=str,help="The path to the base model",required=True)
parser.add_argument('--ea_model_path',type=str,help="The path to the ea model",required=True)

args=parser.parse_args()

model = EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
your_message="What are your views on deforestation?"
conv = get_conversation_template("llama")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
# input_ids=model.tokenizer([your_message]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=15)
output=model.tokenizer.decode(output_ids[0])
print(output)
