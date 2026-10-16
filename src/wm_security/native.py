"""WorldEvolver structured-state adapter. Native registry and extraction prompts.

Only observation serialization/factorization is adapted to exact simulator JSON.
The semantic score update, rendering, extraction, and merge are upstream code.
Gate-free raw predictions are a mechanism diagnostic, not gated deployment results.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys

UPSTREAM=Path(__file__).resolve().parents[2]/'vendor'/'WorldEvolver'/'src'
sys.path.insert(0,str(UPSTREAM))
from world_model.modules.typed_registry import (TypedRegistry, MismatchSample,
    build_extraction_system_msg, build_extraction_user_msg, parse_extracted_json)
from world_model.modules.state_library import EpisodicLibrary
from world_model.modules.delta_card import render_delta_cards
from world_model.prompt_templates import EPISODIC_SEMANTIC_WORLD_MODEL_PROMPT
from .client import parse_json


class NativeMemory:
    def __init__(self, client, env, token_budget=2048, top_k=5):
        self.client=client
        self.env=env
        self.registry=TypedRegistry()
        self.episodic=EpisodicLibrary(env.name)
        self.token_budget=token_budget
        self.top_k=top_k
        self.transitions=[]

    def clone(self):
        result=NativeMemory(self.client, self.env, self.token_budget,self.top_k)
        result.registry=TypedRegistry.from_dict(self.registry.to_dict())
        result.episodic.load_records(self.episodic.records)
        result.transitions=deepcopy(self.transitions)
        return result

    def snapshot(self):
        return {'registry':self.registry.to_dict(),'episodic':deepcopy(self.episodic.records),
                'transitions':deepcopy(self.transitions)}

    def predict(self, state, action, tag='prediction'):
        rules,keys=self.registry.compile_with_keys(token_budget=self.token_budget)
        cards=self.episodic.retrieve_top_k(json.dumps(state),action,self.top_k)
        base=EPISODIC_SEMANTIC_WORLD_MODEL_PROMPT.format(env_name=self.env.name)
        base=base.replace('Output format: a single paragraph starting with "Prediction:" describing the next observation in the style of the environment\'s own text output.',
                          'Output format: a JSON object containing only the next state, with exactly the input state fields. No commentary.')
        system='\n\n'.join([rules,render_delta_cards(cards),base,self.env.schema,
            'If evidence is insufficient, make one best guess. Return your predicted state directly.'])
        response=self.client.json(system, {'current_state':state,'action':action},tag=tag,max_tokens=1024)
        # Some compatible endpoints wrap the requested state in a singleton
        # next_state field. Decode that envelope only when its contents have
        # exactly the expected schema; never fill or change predicted values.
        if (isinstance(response,dict) and set(response)=={'next_state'}
                and isinstance(response['next_state'],dict)
                and set(response['next_state'])==set(state)):
            response=response['next_state']
        if not isinstance(response,dict) or set(response)!=set(state):
            raise ValueError('Invalid state schema: '+repr(response))
        return response, keys, [card.source_index for card in cards]

    def update(self, before, action, after, predicted, keys, tag='update', credit='global'):
        old=self.registry.to_dict()
        mismatch=predicted!=after
        self.episodic.append(task='',state=json.dumps(before),action=action,gold_next_state=json.dumps(after))
        if mismatch:
            if credit=='global':
                self.registry.credit_mismatch(keys)
            elif credit!='no_negative_credit':
                raise ValueError(credit)
            sample=MismatchSample('',json.dumps(before),action,json.dumps(predicted),json.dumps(after))
            raw=self.client.chat([
                {'role':'system','content':build_extraction_system_msg(self.env.name)},
                {'role':'user','content':build_extraction_user_msg([sample],env_name=self.env.name)}],
                tag=tag,max_tokens=1536)
            entries=parse_extracted_json(raw)
            self.registry.merge(entries)
        else:
            self.registry.credit_match(keys)
            entries=[]
        event={'before':before,'action':action,'after':after,'predicted':predicted,
               'mismatch':mismatch,'displayed':sorted(keys),'new_rules':[x.to_dict() for x in entries],
               'registry_before':old,'registry_after':self.registry.to_dict()}
        self.transitions.append(event)
        return event
