
#Zoho Labs Kottarakara: User interface for Sam-Decoding
#Import necessary libraries
import argparse
import gradio as gr 
import torch
import re
from fastchat.model import get_conversation_template
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    GenerationMixin,
    LlamaConfig,
    LlamaTokenizer
)
import webbrowser
from samd import (
    SamdConfig, 
    SamdModel, 
    SamdGenerationConfig,
    DraftModel,
    load_sam
)
#Parse necessary arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--sam_path', type=str, default=None)
parser.add_argument('--samd_n_predicts', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--max_cache_len', type=int, default=2048)
parser.add_argument("--tree_method", type=str, default="eagle2")
parser.add_argument("--tree_model_path", type=str, default=None,required=True)
parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
args = parser.parse_args()

args.dtype = {
    'float16': torch.float16,
    'float32': torch.float32,
}[args.dtype]

#load the model and set to evaluation mode
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    torch_dtype=args.dtype, 
    device_map=args.device,
)
model.eval()

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
    gen = samd_model.stream_generate(**inputs, generation_config=gen_config)
    return gen
        

def user(current_text,chatbot,session_state):
    if chatbot is None:
        chatbot=[]
    pure_history=session_state.get("pure_history",[])
    pure_history+=[[current_text,None]]
    session_state["pure_history"]=pure_history
    return "",chatbot+[[current_text,None]],session_state
def clear(history,session):
    pure_history=[]
    session["pure_history"]=pure_history
    history=pure_history
    return history,session
def regenerate(history,session_state):
    if history is None:
        history=[]
    pure_history=session_state.get("pure_history",[])
    pure_history[-1][-1]=None
    session_state["pure_history"]=pure_history
    history[-1][-1]=None
    return history,session_state
    

def bot(chatbot,session_state):
    pure_history=session_state.get("pure_history",[])
    input_text=pure_history[-1][0]
    inputs = tokenizer(
    input_text, 
    padding=True, 
    return_tensors="pt"
    ).to(args.device)
    
    gen=samd_generate(args, inputs, model, tokenizer)
    coloured_response = ""
    raw_response=""
    colour_data={"tree":"green",
                 "static":"orange",
                 "dynamic":"red"
    }
    all_ids=[]
    previous_output=""
    
    for chunk in gen:
        token_ids = chunk["ids"]
        seqtype = chunk["seqtype"]
        color = colour_data.get(seqtype)
        all_ids.extend(token_ids)

        output_so_far = tokenizer.decode(all_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        token_text = output_so_far[len(previous_output):]
        previous_output = output_so_far

        if len(token_ids) == 1:
            start_token = tokenizer.decode(token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            start_token = start_token.replace("_", " ").strip()
            token_text = re.sub(
                re.escape(start_token),
                f"<span style='color:white'>{start_token}</span>",
                token_text,
                count=1
            )
        else:
            start_token = tokenizer.decode(token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            start_token = start_token.replace("_", " ").strip()
            replacement = (
                f"<span style='color:white'>\\1</span>"
                f"<span style='color:{color}'>\\2</span>"
            )
            token_text = re.sub(
                rf"({re.escape(start_token)})(.*)",
                replacement,
                token_text,
                count=1
            )

        colored_token=token_text
        raw_response+=token_text
        coloured_response += colored_token
        chatbot[-1][1]=coloured_response
        yield chatbot, session_state

    pure_history[-1][1] = raw_response
    session_state["pure_history"] = pure_history



custom_css="""
#regenerate_button,
#clear_button,
#stop_button,
#send_button {
    background-color: #FFFFFF;
    color: black;
    font-size: 20px;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: bold;
}
#regenerate_button:hover,
#clear_button:hover,
#stop_button:hover,
#send_button:hover {
    background-color: #FFA500;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    <h1 style="text-align:center; color: orange;">SAM-CHATBOT</h1>
    """)
    gs=gr.State({"pure_history": []})
    chatbot=gr.Chatbot(height=600, show_label=False)
    msg=gr.Textbox(label="Input")
    with gr.Row():
        send_button=gr.Button("Send",elem_id="send_button")
        stop_button=gr.Button("Stop",elem_id="stop_button")
        regenerate_button=gr.Button("Regenerate",elem_id="regenerate_button")
        clear_button=gr.Button("Clear",elem_id="clear_button")
    with gr.Row():
            gr.Markdown("""
        <h3 style="text-align:center; color: white;">⚪-Verifier 🟢-EAGLE  🟠-Static 🔴-Dynamic</h3>
        """)
    enter_event=msg.submit(user,[msg,chatbot,gs],[msg,chatbot,gs]).then(bot,[chatbot,gs],[chatbot,gs])
    send_event=send_button.click(user,[msg,chatbot,gs],[msg,chatbot,gs]).then(bot,[chatbot,gs],[chatbot,gs])
    clear_event=clear_button.click(clear,[chatbot,gs],[chatbot,gs],cancels=[enter_event,send_event])
    regenerate_event=regenerate_button.click(regenerate,[chatbot,gs],[chatbot,gs]).then(bot,[chatbot,gs],[chatbot,gs])
    stop_event=stop_button.click(None,None,None,cancels=[send_event,enter_event])


demo.launch(server_name="0.0.0.0", server_port=5500)

    
