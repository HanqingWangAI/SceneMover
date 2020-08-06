import numpy as np 
import pickle
import random
from copy import deepcopy

from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly as classic
from env import ENV_ablation_wo_base as base
from env import ENV_ablation_wo_multi as multi
from env import ENV_ablation_wo_repetition as repetition

ENV = {'classic': classic,
        'base': base,
        'multi': multi,
        'repetition': repetition
}

class DataIO(object):
    def __init__(self, dataset=None, batch_size = 64, max_num = 25, action_type = 5,env_type='classic'):
        if dataset is not None:
            with open(dataset,'rb') as fp:
                self.data = pickle.load(fp)
            random.seed(0)
            # random.shuffle(self.data)
            print(dataset,'len',len(self.data))
            self.envs = []
            self.batch_size = min(batch_size,len(self.data))
            self.idx = 0
            self.max_num = max_num
            self.action_type = action_type
            self.action_space = self.max_num * self.action_type
            for i in range(self.batch_size):
                self.envs.append(ENV[env_type](size=(64,64)))
        else:
            self.data = []

        
    
    def _next_batch(self):
        batch = self.data[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        
        if self.idx >= len(self.data):
            self.idx = self.idx % len(self.data)
            batch += self.data[:self.idx]
        
        self.batch = batch

    
    def reset(self,keep=False):
        if not keep:
            self._next_batch()
            
        batch_actions = []
        for i,item in enumerate(self.batch):
            pos = item['pos']
            target = item['target']
            shape = item['shape']
            cstate = item['cstate']
            tstate = item['tstate']
            wall = item['wall']
            batch_actions.append(item['actions'])
            self.envs[i].setmap(pos, target, shape, cstate, tstate, wall)
        
        return batch_actions

    def make_action_logits(self, logits):
        rewards = []
        actions = []
        for i,lo in enumerate(logits):
            while True:
                a = lo.argmax()
                item = int(a / self.action_type)
                d = int(a % self.action_type)
                reward, flag = self.envs[i].move(item, d)
                if flag == -1:
                    lo[a] = -np.inf
                    continue
                # print(flag)
                
                rewards.append((reward, flag))
                actions.append(a)
                break
        
        return rewards, actions
    
    def make_action(self, actions):
        rewards = []
        for i,a in enumerate(actions):
            if a == -1:
                rewards.append((0,-1))
                continue

            item = int(a / self.action_type)
            d = int(a % self.action_type)
            reward, flag = self.envs[i].move(item, d)
            # print(flag)
            # assert flag != -1
            rewards.append((reward, flag))
        
        return rewards


    def get_state(self,tf=True):
        state = []
        tag = []
        for i,env in enumerate(self.envs):
            state.append(env.getstate_1())
            tag.append(env.getfinished())
        
        if tf:
            state = np.array(state)
        else:
            state = np.array(state).transpose(0,3,1,2) # for pytorch

        # print(state.shape)
        return state, np.array(tag)

    def __deepcopy__(self):
        res = DataIO()
        res.envs = []
        for env in self.envs:
            res.envs.append(deepcopy(env))
        
        res.action_type = self.action_type
        res.action_space = self.action_space
        res.max_num = self.max_num
        res.batch_size = self.batch_size
        res.idx = self.idx
        
        return res


        