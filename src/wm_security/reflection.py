"""WorldMind-prompt-derived process learning for structured simulator states.

Uses upstream reflection prompts verbatim. Exact state comparison replaces the
multimodal discriminator; small pilots read all lessons instead of embedding
retrieval. This is explicitly an adapter, not an original-benchmark reproduction.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from .native import NativeMemory,TypedRegistry
from world_model.modules.typed_registry import RegistryEntry
from .client import parse_json

path=Path(__file__).resolve().parents[2]/'vendor/WorldMind_selected/prompts.py'
spec=importlib.util.spec_from_file_location('worldmind_source_prompts',path)
prompts=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompts)


class ReflectionMemory(NativeMemory):
    def clone(self):
        m=ReflectionMemory(self.client,self.env,self.token_budget,self.top_k)
        m.registry=TypedRegistry.from_dict(self.registry.to_dict())
        m.episodic.load_records(self.episodic.records)
        m.transitions=deepcopy(self.transitions)
        return m

    def predict(self,state,action,tag='prediction'):
        lessons=[e.text for e in self.registry.entries.values()]
        context=prompts.PROCESS_EXPERIENCE_RETRIEVAL_CONTEXT.format(knowledge='\n'.join(lessons)) if lessons else ''
        system=('You are a world model. Predict the state after the given action using learned experience '
                'and the current state. Output only a JSON object with exactly the same state fields. '
                'If evidence is insufficient, make one best guess.\n'+self.env.schema+'\n'+context)
        pred=self.client.json(system,{'current_state':state,'action':action},tag=tag,max_tokens=1024)
        if not isinstance(pred,dict) or set(pred)!=set(state):raise ValueError('Invalid state schema')
        return pred,set(self.registry.entries),[]

    def update(self,before,action,after,predicted,keys,tag='update',credit='global'):
        old=self.registry.to_dict();mismatch=predicted!=after;entries=[]
        self.episodic.append(task='',state=json.dumps(before),action=action,gold_next_state=json.dumps(after))
        if mismatch:
            history=[{k:e[k] for k in ['before','action','after']} for e in self.transitions[-4:]]
            user=prompts.REFLECTOR_USER_PROMPT.format(human_instruction='Learn the dynamics of '+self.env.name,
                action_history=json.dumps(history),state_before=json.dumps(before),action_description=action,
                env_feedback=json.dumps(after),predicted_state=json.dumps(predicted),state_after=json.dumps(after))
            raw=self.client.chat([{'role':'system','content':prompts.REFLECTOR_SYSTEM_PROMPT},
                                  {'role':'user','content':user}],tag=tag,max_tokens=1536)
            result=parse_json(raw)
            for text in result.get('experience_entry',[]):
                if isinstance(text,str):entries.append(RegistryEntry('lesson.'+str(len(self.registry)+len(entries)),text))
            self.registry.merge(entries)
        event={'before':before,'action':action,'after':after,'predicted':predicted,'mismatch':mismatch,
               'displayed':sorted(keys),'new_rules':[x.to_dict() for x in entries],
               'registry_before':old,'registry_after':self.registry.to_dict()}
        self.transitions.append(event)
        return event
