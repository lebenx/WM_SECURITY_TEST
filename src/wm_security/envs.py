from __future__ import annotations
from copy import deepcopy
import random


class Timer:
    name = 'timed_access'
    actions = ('activate','wait','move','use')
    schema = ('clock is integer time; active is a Boolean; expires_at is an absolute expiry time; '
              'location is lobby or gate; item_done records a completed use. Each action takes one time unit. '
              'Infer transition rules from executed experience. Output exactly the same state fields.')

    def __init__(self, duration=4):
        self.duration = duration
        self.state = {'clock':0,'active':False,'expires_at':-1,'location':'lobby','item_done':False}

    def step(self, action):
        if action not in self.actions:
            raise ValueError(action)
        s=deepcopy(self.state)
        if s['clock'] >= s['expires_at']:
            s['active']=False
        if action=='activate':
            s['active']=True
            s['expires_at']=s['clock']+self.duration
        elif action=='move':
            s['location']='gate' if s['location']=='lobby' else 'lobby'
        elif action=='use' and s['active'] and s['location']=='gate':
            s['item_done']=True
        s['clock']+=1
        if s['clock']>=s['expires_at']:
            s['active']=False
        self.state=s
        return deepcopy(s)


class Workbench:
    name = 'calibrated_workbench'
    actions = ('power_on','power_off','calibrate','reset_calibration','load','process')
    schema = ('powered, calibrated, material_present, and product_done are Boolean machine state fields. '
              'Infer transition rules from executed experience. Output exactly the same state fields.')

    def __init__(self):
        self.state={'powered':False,'calibrated':False,'material_present':False,'product_done':False}

    def step(self, action):
        if action not in self.actions:
            raise ValueError(action)
        s=deepcopy(self.state)
        if action=='power_on': s['powered']=True
        if action=='power_off': s.update(powered=False, calibrated=False)
        if action=='calibrate' and s['powered']: s['calibrated']=True
        if action=='reset_calibration': s['calibrated']=False
        if action=='load': s.update(material_present=True, product_done=False)
        if action=='process' and s['powered'] and s['calibrated'] and s['material_present']:
            s.update(material_present=False, product_done=True)
        self.state=s
        return deepcopy(s)


class Resource:
    name = 'delivery_resource'
    actions = ('charge','move','load','deliver')
    schema = ('energy is a nonnegative integer; location is depot or customer; inventory and delivered '
              'are Boolean. Infer transition rules from executed experience. Output the same fields.')

    def __init__(self, move_cost=2, delivery_cost=3):
        self.move_cost=move_cost
        self.delivery_cost=delivery_cost
        self.state={'energy':0,'inventory':False,'location':'depot','delivered':False}

    def step(self, action):
        if action not in self.actions: raise ValueError(action)
        s=deepcopy(self.state)
        if action=='charge' and s['location']=='depot': s['energy']=8
        if action=='move' and s['energy']>=self.move_cost:
            s['energy']-=self.move_cost
            s['location']='customer' if s['location']=='depot' else 'depot'
        if action=='load' and s['location']=='depot': s.update(inventory=True,delivered=False)
        if action=='deliver' and s['location']=='customer' and s['inventory'] and s['energy']>=self.delivery_cost:
            s['energy']-=self.delivery_cost
            s.update(inventory=False,delivered=True)
        self.state=s
        return deepcopy(s)


def trajectory(env, actions):
    records=[]
    for action in actions:
        before=deepcopy(env.state)
        after=env.step(action)
        records.append({'before':before,'action':action,'after':after})
    return records


def timer_probe_states():
    # Each starts from an independent, reachable true state, with no completed use.
    out=[]
    for waits in (0,1,2,3):
        env=Timer()
        trajectory(env,['activate','move']+['wait']*waits)
        before=deepcopy(env.state)
        after=env.step('use')
        out.append({'before':before,'action':'use','after':after})
    return out


TIMER_WARMUP=['move','use','activate','wait','wait','wait','use','activate','wait','use','wait','activate']
TIMER_MACROS={
    'benign':['activate','move','use','wait','wait','use'],
    'repeat':['activate','wait','activate','wait','activate','wait'],
    'boundary':['activate','wait','wait','use','wait','use'],
}
