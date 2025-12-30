import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm
import networkx as nx
import os
from pathlib import Path
from transformers import AutoTokenizer

class StaticSAM:
   
    @dataclass
    class SAMState:
        next: dict[int, int]
        link: int
        length: int
        min_endpos: int

    def __init__(self, n_predicts: int = 40):
        self.n_predicts = n_predicts
        print("The n predicts are",self.n_predicts)
        self.states: List[StaticSAM.SAMState] = [StaticSAM.SAMState(next={}, link=-1, length=0, min_endpos=0)]
        self.input_ids: List[int] = [-1]
        self.last = 0
        self.max_length = 0
        self.counter=0
        # self.tokenizer=AutoTokenizer.from_pretrained("")
        self.exec_counter=0
        # params needed to be reset for each query
        self.cur_index = 0
        self.cur_length = 0
        self.currpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.graphdir = os.path.join(self.currpath, "Graphs")#Directory to store the graphs and json
        if not os.path.exists(self.graphdir):
            os.mkdir(self.graphdir)

        Path(self.graphdir,"Image").mkdir(exist_ok=True)

        self.image_path=os.path.join(self.graphdir,"Image")
        if self.counter==0:
            if os.path.exists(self.image_path):
                for items in os.listdir(self.image_path):
                    item_path=os.path.join(self.image_path,items)
                    os.remove(item_path)

    
    def reset(self):
        self.cur_index = 0
        self.cur_length = 0

    def add_batch_tokens(self, batch_tokens: List[List[int]], eos_token: int, verbose: bool):
        for tokens in tqdm(batch_tokens, desc="build sam...", disable=not verbose):
            self.add_tokens(tokens)
            if tokens[-1] != eos_token:
                self.add_tokens([eos_token])

    @staticmethod
    def build(
        batch_tokens: List[List[int]], 
        eos_token: int,
        verbose: bool =True
    ):
        sam = StaticSAM()
        sam.add_batch_tokens(batch_tokens, eos_token, verbose)
        return sam
    
    def expand_state(self, state: SAMState):
        new_index = len(self.states)
        self.states.append(state)
        return new_index

    def add_state(self, token: int):
        self.max_length += 1
        cur = self.expand_state(
            StaticSAM.SAMState(
                next={}, link=-1, 
                length=self.max_length, 
                min_endpos=self.max_length, 
            )
        )
        p = self.last 
        while p != -1 and token not in self.states[p].next:
            self.states[p].next[token] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                clone = self.expand_state(deepcopy(self.states[q]))
                self.states[clone].length = self.states[p].length + 1
                while p != -1 and self.states[p].next[token] == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[cur].link = clone
        self.last = cur

  
    def transfer_state(self, index: int, length: int, token: int,is_infer=None,step=None,counter=None):
        if counter!=None:
            self.counter=counter
        while index != 0 and token not in self.states[index].next:#executed untill the condition fails
            # if is_infer==True:
            #     self.dfa_link_graph(index,step)
            index = self.states[index].link
            length = self.states[index].length
        if token in self.states[index].next:#If correct, this gets executed only one time
            # if is_infer==True:
            #     self.dfa_next_graph(index,token,step)
            index = self.states[index].next[token]
            length += 1
        else:
            index = length = 0
        
        if is_infer==True:
            return index, length,self.counter
        else:
            return index, length
    
    
    def transfer_cur_state(self, token: int):
        self.cur_index, self.cur_length = \
            self.transfer_state(self.cur_index, self.cur_length, token)
   
    def add_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)
            self.add_state(token)
        self.input_ids.extend(tokens)
    
    def transfer_tokens(self, tokens: List[int]):
        for token in tokens:
            self.transfer_cur_state(token)

    def lookup(self, token: int,step,counter):
        index, length,counter = \
            self.transfer_state(self.cur_index, self.cur_length, token, is_infer=True,step=step,counter=counter)
        return index, length, counter

    def to_anc(self, index: int):
        if index != 0:
            length_to_end = self.max_length - self.states[index].min_endpos
            while self.states[index].link != 0 and self.n_predicts > length_to_end:
                index = self.states[index].link
                length_to_end = self.max_length - self.states[index].min_endpos
        return index

    def gen_draft(self, index: int, start_token: int):
        # index = self.to_anc(index)
        endpos = self.states[index].min_endpos
        pred_ids = [start_token] + self.input_ids[endpos + 1:endpos + self.n_predicts]
        if len(pred_ids) < self.n_predicts:
            pred_ids.extend([0] * (self.n_predicts - len(pred_ids)))
        return pred_ids
    
    def dfa_next_graph(self,index,token,step):
        current_index=index
        next_index=self.states[index].next[token]
        self.draw((current_index,next_index),step,edge_color="black",token=token)
    def dfa_link_graph(self,index,step):
        current_index=index
        link_index=self.states[index].link
        self.draw((current_index,link_index),step,edge_color="Red")
    
    def sanitize(self,token_text):
        if token_text=="\n":
            token_text="\\\n"
        elif token_text == " ":
            token_text = "[space]"   
        elif token_text == "\t":
            token_text = "\\\t"        
        elif token_text.strip() == "":
            token_text = "[blank]"
        else:
            token_text=token_text
        return token_text
    
    #Zoho Labs Kottarakara:Visualize the dfa transitions for Static-SAM
    
    # def draw(self,transition,step,edge_color,token=None):
    #     G=nx.DiGraph()
    #     color="lightblue"
    #     if edge_color=="black":#next condition
    #         for index in transition: 
    #             if index==0:
    #                 token_text="<ROOT>"
    #             else:
    #                 token_id=self.input_ids[self.states[index].min_endpos]
    #                 token_text=self.tokenizer.decode([token_id])
    #             token_text=str(self.sanitize(token_text))
    #             G.add_node(index,label=token_text,style="filled",fillcolor=color,fontweight="bold", fontsize="20", fontcolor="black")
    #         edge_label=self.tokenizer.decode([token])
    #         edge_label=str(self.sanitize(edge_label))
    #         G.add_edge(transition[0],transition[1],label=edge_label,color=edge_color)
    #         pydot_graph = nx.nx_pydot.to_pydot(G)
    #         pydot_graph.set_label("This graph curresponds to next in static")
    #         pydot_graph.set("labelloc", "t")   # top of the graph
    #         pydot_graph.set("fontsize", "16")
    #         pydot_graph.set("fontcolor", "black")
    #         pydot_graph.set_size('"12,10!"')
    #         pydot_graph.set_ratio('fill')
    #         pydot_graph.set("dpi", "600")
    #         image_path=os.path.join(self.graphdir,"Image")
    #         image_name=f"{step}_image_{self.counter}_.png"
    #         image_absolute_path=os.path.join(image_path,image_name)
    #         pydot_graph.write_png(image_absolute_path)
    #         self.counter+=1
    #     else:#link condiition
    #         for index in transition:
    #             if index==-1:
    #                 token_text="<IMG>"
    #             elif index==0:
    #                 token_text="<ROOT>"
    #             else:
    #                 token_id=self.input_ids[self.states[index].min_endpos]
    #                 token_text=self.tokenizer.decode([token_id])
    #                 token_text=self.sanitize(token_text)

    #             G.add_node(index,label=token_text,style="filled",fillcolor=color,fontweight="bold", fontsize="20", fontcolor="black")
    #         G.add_edge(str(transition[0]),str(transition[1]),color=edge_color)
    #         pydot_graph = nx.nx_pydot.to_pydot(G)
    #         pydot_graph.set_label("The graph curresponds to link in static")
    #         pydot_graph.set("labelloc", "t")   # top of the graph
    #         pydot_graph.set("fontsize", "16")
    #         pydot_graph.set("fontcolor", "black")
    #         pydot_graph.set_size('"12,10!"')
    #         pydot_graph.set_ratio('fill')
    #         pydot_graph.set("dpi", "600")
    #         image_path=os.path.join(self.graphdir,"Image")
    #         image_name=f"{step}_image_{self.counter}_.png"
    #         image_absolute_path=os.path.join(image_path,image_name)
    #         pydot_graph.write_png(image_absolute_path)          
    #         self.counter+=1


class NullStaticSAM(StaticSAM):
    
    def __init__(self, n_predicts = 40):
        super().__init__(n_predicts)
    
    def transfer_tokens(self, tokens):
        pass
    
    def gen_draft(self, index, start_token):
        return -1, -1