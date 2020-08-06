import os
import torch
import torch.nn as nn
from collections import defaultdict

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
batch_size = 64
gamma = 0.95

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    a = torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    b = (torch.LongTensor(length) - 1)
    c = b.unsqueeze(1)
    d = (a > c).cuda()

    mask = d
    return mask

class Encoder(nn.Module):
    def __init__(self, state_size=[64,64,2],hidden_size=512,max_num=25, action_type=5,feature_size=4096):
        super(Encoder, self).__init__()
        self.state_size = state_size
        self.action_space = max_num * action_type
        self.max_num = max_num
        self.action_type = action_type
        
        self.flat = nn.Flatten()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d( in_channels=state_size[-1],
                                out_channels = 64,
                                kernel_size=5,
                                stride=2,
                                padding=2)

        self.conv2 = nn.Conv2d( in_channels=64,
                                out_channels = 128,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        
        self.conv3 = nn.Conv2d( in_channels=128,
                                out_channels = 256,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        
        self.conv4 = nn.Conv2d( in_channels=256,
                                out_channels = 256,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        
        self.lstm = nn.LSTMCell(feature_size+self.action_space+self.max_num, hidden_size)
        self.linear = nn.Linear(hidden_size, self.action_space, bias=False)
        
    def forward(self, x, pre_action, finish_tag, h_0, c_0):
        # x = x.float()
        c1 = self.lrelu(self.conv1(x))  # 32 x 32 x 64
        c2 = self.lrelu(self.conv2(c1)) # 16 x 16 x 128
        c3 = self.lrelu(self.conv3(c2)) # 8  x  8 x 256
        c4 = self.conv4(c3) # 4  x  4 x 256
        flat = self.flat(c4) # 4096
        feature = torch.cat([flat,pre_action,finish_tag],-1)
        h_1, c_1 = self.lstm(feature,(h_0, c_0))
        logits = self.linear(h_1)
        # print('c1',c1.shape)
        # print('c2',c2.shape)
        # print('c3',c3.shape)
        # print('c4',c4.shape)
        # print('out',out.shape)
        return logits, h_1, c_1

class Critic(nn.Module):
    def __init__(self, hidden_size=512):
        super(Critic, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        
        
    def forward(self, x):
        x = self.lrelu(self.linear1(x))
        res = self.linear2(x).squeeze()
        return res


class Agent(object):
    def __init__(self,state_size=[64,64,51],hidden_size=512, max_num=25, action_type=5,feature_size=4096, episode_len = 80):
        self.encoder = Encoder(state_size, hidden_size, max_num, action_type, feature_size).cuda()
        self.critic = Critic(hidden_size=hidden_size).cuda()
        self.encoder_opt = torch.optim.RMSprop(self.encoder.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.RMSprop(self.critic.parameters(), lr=1e-4)

        self.models = [self.encoder, self.critic]
        self.optimizers = [self.encoder_opt, self.critic_opt]

        
        self.hidden_size = hidden_size
        self.max_num = max_num
        self.action_type = action_type
        self.action_space = action_type * max_num
        self.episode_len = episode_len
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)


    def teacher_action(self, t, actions):
        res = []
        for i, a in enumerate(actions):
            if t >= len(a):
                res.append(-1)
            else:
                it, d = a[t]
                res.append(it*self.action_type+d)

        return torch.from_numpy(np.array(res)).cuda()

    def rollout_train(self, env, train_il=1):
        batch_size = env.batch_size
        actions = env.reset()

        c1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()
        h1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()

        ended = np.array([False] * batch_size)

        pre_action = np.zeros([batch_size, self.action_space])
        pre_action = torch.from_numpy(pre_action).float().cuda()

        
        loss = 0
        cnt = 0
        for t in range(self.episode_len):
            state, tag = env.get_state()
            state = torch.from_numpy(state).float().cuda()
            tag = torch.from_numpy(tag).float().cuda()

            logits, h1,c1 = self.encoder(state, pre_action, tag, h1, c1)

            teacher = self.teacher_action(t, actions)
            loss += (self.criterion(logits, teacher) * (1.0 - torch.from_numpy(ended).float().cuda())).sum()
            cnt += (1.0 - torch.from_numpy(ended).float().cuda()).sum().item()

            a_t_cpu = teacher.cpu().numpy()

            pre_action = np.zeros([batch_size, self.action_space])
            for i,a_ in enumerate(a_t_cpu):
                pre_action[i,a_] = 1
            pre_action = torch.from_numpy(pre_action).float().cuda()

            rewards = env.make_action(a_t_cpu)
            done = np.array([item[1] for item in rewards])
            
            ended = np.logical_or(ended, (done == 1))

            if ended.all():
                break
        # print(t)
        self.loss += train_il * loss / cnt
        self.logs['loss'] = self.loss.item()
    
    def rollout_train_rl(self, env, train_rl=0.5,keep=True):
        batch_size = env.batch_size
        actions = env.reset(keep=keep)

        c1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()
        h1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()

        ended = np.array([False] * batch_size)

        pre_action = np.zeros([batch_size, self.action_space])
        pre_action = torch.from_numpy(pre_action).float().cuda()

        cnt = 0
        loss = 0
        rewards_list = []
        mask_list = []
        # logits_list = []
        log_prob_logit = []
        hidden_states = []
        critic_loss = 0.
        actor_loss = 0.
        for t in range(50):
            state, tag = env.get_state()
            state = torch.from_numpy(state).float().cuda()
            tag = torch.from_numpy(tag).float().cuda()

            logits, h1, c1 = self.encoder(state, pre_action, tag, h1, c1)

            
            
            hidden_states.append(h1)
            logits_cpu = logits.detach().cpu().numpy()
            rewards, a_t_cpu = env.make_action_logits(logits_cpu)

            pre_action = np.zeros([batch_size, self.action_space])
            for i,a_ in enumerate(a_t_cpu):
                pre_action[i,a_] = 1
            pre_action = torch.from_numpy(pre_action).float().cuda()

            # logits_list.append((logits * pre_action).sum(-1))
            log_prob_logit.append((torch.log(torch.softmax(logits, -1))*pre_action).sum(-1))

            done = np.array([item[1] for item in rewards])
            reward = np.array([item[0] for item in rewards])

            mask = np.ones(batch_size, np.float32)
            for i,e in enumerate(ended):
                if e:
                    mask[i] = 0
                    reward[i] = 0
                
            mask_list.append(mask)
            rewards_list.append(reward)

            
            ended = np.logical_or(ended, (done == 1))

            if ended.all():
                break
        

        
        state, tag = env.get_state()
        state = torch.from_numpy(state).float().cuda()
        tag = torch.from_numpy(tag).float().cuda()

        # # last_value, _, _ = self.encoder(state, pre_action, tag, h1, c1)
        # last_value, _ = last_value.max(-1)
        last_value = self.critic(h1)
        last_value = last_value.detach().cpu().numpy()
        discount_reward = np.zeros(batch_size, np.float32)
        
        for i in range(batch_size):
            if not ended[i]:
                discount_reward[i] = last_value[i]
        
        length = len(rewards_list)
        # print(max([r.max() for r in rewards_list]))
        # print(min([r.min() for r in rewards_list]))
        
        for t in range(length-1, -1, -1):
            discount_reward = rewards_list[t] + gamma * discount_reward
            mask = torch.from_numpy(mask_list[t]).cuda()
            r = torch.from_numpy(discount_reward).cuda()
            v = self.critic(hidden_states[t])
            a = (r-v).detach()
            critic_loss += ( ((r - v) ** 2) * mask).sum() # critic loss
            actor_loss += (-log_prob_logit[t] * a * mask).sum()
            cnt += mask.sum()

        critic_loss /= cnt
        actor_loss /= cnt
        # print(t)
        self.loss += train_rl * (critic_loss + actor_loss)
        self.logs['loss_rl'] = self.loss.item()
        self.logs['critic_loss'] = critic_loss.item()
        self.logs['actor_loss'] = actor_loss.item()

    def rollout_train_rl_v2(self, env, train_rl=0.0001,keep=True):
        batch_size = env.batch_size
        actions = env.reset(keep=keep)

        c1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()
        h1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()

        ended = np.array([False] * batch_size)

        pre_action = np.zeros([batch_size, self.action_space])
        pre_action = torch.from_numpy(pre_action).float().cuda()

        cnt = 0
        loss = 0
        rewards_list = []
        mask_list = []
        logits_list = []
        for t in range(50):
            state, tag = env.get_state()
            state = torch.from_numpy(state).float().cuda()
            tag = torch.from_numpy(tag).float().cuda()

            logits, h1, c1 = self.encoder(state, pre_action, tag, h1, c1)
            

            logits_cpu = logits.detach().cpu().numpy()
            rewards, a_t_cpu = env.make_action_logits(logits_cpu)

            pre_action = np.zeros([batch_size, self.action_space])
            for i,a_ in enumerate(a_t_cpu):
                pre_action[i,a_] = 1
            pre_action = torch.from_numpy(pre_action).float().cuda()

            logits_list.append((logits * pre_action).sum(-1))


            done = np.array([item[1] for item in rewards])
            reward = np.array([item[0] for item in rewards])

            mask = np.ones(batch_size, np.float32)
            for i,e in enumerate(ended):
                if e:
                    mask[i] = 0
                    reward[i] = 0
                
            mask_list.append(mask)
            rewards_list.append(reward)

            
            ended = np.logical_or(ended, (done == 1))

            if ended.all():
                break
        

        
        state, tag = env.get_state()
        state = torch.from_numpy(state).float().cuda()
        tag = torch.from_numpy(tag).float().cuda()

        last_value, _, _ = self.encoder(state, pre_action, tag, h1, c1)
        last_value, _ = last_value.max(-1)
        last_value = last_value.detach().cpu().numpy()
        discount_reward = np.zeros(batch_size, np.float32)
        
        for i in range(batch_size):
            if not ended[i]:
                discount_reward[i] = last_value[i]
        
        length = len(rewards_list)
        
        
        for t in range(length-1, -1, -1):
            discount_reward = rewards_list[t] + gamma * discount_reward
            mask = torch.from_numpy(mask_list[t]).cuda()
            v = torch.from_numpy(discount_reward).cuda()
            loss += (torch.abs((v - logits_list[t])) * mask).sum()
            cnt += mask.sum()

        # print(t)
        self.loss += train_rl * loss / cnt
        self.logs['loss_rl'] = self.loss.item()


    def rollout_test(self, env):
        batch_size = env.batch_size
        actions = env.reset()

        c1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()
        h1 = torch.from_numpy(np.zeros([batch_size,self.hidden_size])).float().cuda()

        ended = np.array([False] * batch_size)

        pre_action = np.zeros([batch_size, self.action_space])
        pre_action = torch.from_numpy(pre_action).float().cuda()

        traj_length = np.zeros(batch_size).astype(np.int32)

        for t in range(self.episode_len):
            state, tag = env.get_state()
            state = torch.from_numpy(state).float().cuda()
            tag = torch.from_numpy(tag).float().cuda()

            logits, h1,c1 = self.encoder(state, pre_action, tag, h1, c1)
            
            logits_cpu = logits.detach().cpu().numpy()
            rewards, a_t_cpu = env.make_action_logits(logits_cpu)

            pre_action = np.zeros([batch_size, self.action_space])
            for i,a_ in enumerate(a_t_cpu):
                pre_action[i,a_] = 1
            pre_action = torch.from_numpy(pre_action).float().cuda()

            
            done = np.array([item[1] for item in rewards])

            traj_length += (t+1) * np.logical_and(ended == 0, (done == 1))
            
            ended = np.logical_or(ended, (done == 1))

            if ended.all():
                break
        traj_length += self.episode_len * (ended == 0)

        self.logs['success_rate'].append(ended.sum()/batch_size)
        self.logs['average_step'].append(traj_length.mean())
        
    def train_IL(self, env):
        self.logs = defaultdict(list)
        self.loss = 0.
        for model in self.models:
            model.train()
        
        for optim in self.optimizers:
            optim.zero_grad()


        self.rollout_train(env)

        self.loss.backward()

        for optim in self.optimizers:
            optim.step()

    def train_RL(self, env, keep=True):
        self.logs = defaultdict(list)
        self.loss = 0.
        for model in self.models:
            model.train()
        
        for optim in self.optimizers:
            optim.zero_grad()

        self.rollout_train_rl(env,keep=keep)

        self.loss.backward()

        for optim in self.optimizers:
            optim.step()

    def test(self, env, iters=None):
        self.logs = defaultdict(list)
        env.idx = 0
        for model in self.models:
            model.eval()
        
        if iters is None:
            iters = int(len(env.data) / env.batch_size)+1

        with torch.no_grad():
            for i in range(iters):
                self.rollout_test(env)
        
    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_opt),
                    ("critic", self.critic, self.critic_opt)
                    
                     ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path, part=False):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            optimizer.load_state_dict(states[name]['optimizer'])
        if part:
            all_tuple = [("encoder", self.encoder, self.encoder_opt)
                      
                        ]
        else:
            all_tuple = [("encoder", self.encoder, self.encoder_opt)
                        ]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1