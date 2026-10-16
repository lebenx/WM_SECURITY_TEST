from __future__ import annotations
import json
from pathlib import Path
import threading
import time
import uuid
from .client import Client


class LocalClient(Client):
    def __init__(self, model='Qwen/Qwen2.5-Coder-7B-Instruct',log_dir='runs/local',**kwargs):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(8)
        self.torch=torch
        self.model=model
        self.path=Path(log_dir)/'calls.jsonl';self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        self.generate_lock=threading.Lock()
        self.calls=0
        self.max_calls=kwargs.get('max_calls',500)
        cached=Path('/home/xhj/.cache/huggingface/hub')/('models--'+model.replace('/','--'))/'snapshots'
        snapshots=sorted(cached.glob('*'))
        model_path=str(snapshots[-1]) if snapshots else model
        self.revision=snapshots[-1].name if snapshots else None
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.network=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
            dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa')
        self.network.eval()
        print(json.dumps({'local_loaded':model,'revision':self.revision}),flush=True)

    def chat(self,messages,max_tokens=1536,temperature=0,tag='',json_mode=False):
        with self.generate_lock:
            if self.calls>=self.max_calls: raise RuntimeError('Local call ceiling reached')
            self.calls+=1
            started=time.monotonic()
            rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                                         enable_thinking=False)
            inputs=self.tokenizer(rendered,return_tensors='pt').to(self.network.device)
            with self.torch.inference_mode():
                outputs=self.network.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
            generated=outputs[0,inputs['input_ids'].shape[1]:]
            content=self.tokenizer.decode(generated,skip_special_tokens=True)
            self.log({'id':str(uuid.uuid4()),'tag':tag,'time':time.time(),'endpoint':'local',
                'request':{'model':self.model,'messages':messages,'max_tokens':max_tokens,'temperature':0},
                'response':{'model':self.model,'revision':self.revision,
                  'choices':[{'message':{'role':'assistant','content':content},
                              'finish_reason':'length' if len(generated)==max_tokens else 'stop'}],
                  'usage':{'prompt_tokens':inputs['input_ids'].shape[1],
                           'completion_tokens':len(generated),
                           'total_tokens':inputs['input_ids'].shape[1]+len(generated)}},
                'latency_s':time.monotonic()-started})
            return content
