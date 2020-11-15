import tensorflow as tf
import numpy as np
import os
from copy import deepcopy
from collections import deque
import time
import pickle
import random

import threading



def train_net18_2_onehotchannel(teacher_force=True, batch_size = 64, tensorboard_path = "snap/191228_1/"):
    from utils.dataio import DataIO
    from utils.lstm import DQNetwork18_2, DQNetwork18_eval
    from tensorboardX import SummaryWriter
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 64
    # frame_num = 5
    obj_num = 25
    action_space = 5
    start_to_train = 100
    seq_len = 50
    state_size = [map_size, map_size, obj_num*2+1]
    action_size = obj_num * action_space
    dim_h = 256

    def teacher_action(t, actions):
        res = []
        for i, a in enumerate(actions):
            if t >= len(a):
                res.append(-1)
            else:
                it, d = a[t]
                res.append(it*action_space+d)

        return np.array(res)

    # weight_path = "snap/191228_1/"
    weight_path = tensorboard_path


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net_train = DQNetwork18_2(batch_size=batch_size,seq_len=seq_len,state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    net = DQNetwork18_eval(batch_size=1,seq_len=1,state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)

    
    '''
        Setup env
    '''
    env = DataIO('../data/train_IL.pkl',batch_size)
    env_tests = [DataIO('../data/test_%d.pkl'%((i+1)*5),batch_size) for i in range(4)]
    env_test_train = DataIO('../data/train_IL.pkl',batch_size)
    
    # Setup TensorBoard Writer
    # writer = tf.summary.FileWriter(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    

    ## Losses
    # tf.summary.scalar("Loss", net_train.loss)


    # write_op = tf.summary.merge_all()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    
    saver = tf.train.Saver(var_list,max_to_keep=100)
    

    # gpu = tf.config.experimental.list_logical_devices(device_type='GPU')[0]
    # tf.config.experimental.set_memory_growth(gpu, True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())


    decay_step = 0

    started_from = 0

    z_state = np.zeros([batch_size,dim_h])
    h_state = (z_state,z_state)

    init_state = (z_state,z_state)
    
    for step in range(started_from, total_episodes):
        ended = np.array([False] * batch_size)
        actions = env.reset()

        # data for the time steps
        state_list = []
        finish_tag_list = []
   

        masks = np.zeros([batch_size, seq_len])
        action_mask = np.zeros([batch_size, seq_len, action_size])
        Q_target = np.zeros([batch_size, seq_len])
        traj_length = np.zeros(batch_size).astype(np.int32)

        h_state = init_state

        
        rewards_list = [] # seq_len x bs



        for t in range(seq_len):
            # print(t)
            state, tag = env.get_state()
            state_list.append(state) # len x bs x N x M x C
            finish_tag_list.append(tag) # len x bs x K
 
            
            ## EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            ## First we randomize a number
            exp_exp_tradeoff = np.random.rand()

            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

            logits, h_state = sess.run([net.output, net.state_out], feed_dict = {net.inputs_: np.expand_dims(state,1), net.finish_tag: np.expand_dims(tag,1), net.state_in:h_state})


            # TODO(me) change this to be a naive policy
            if (explore_probability > exp_exp_tradeoff):
                logits = np.random.rand(batch_size, action_size)
                
            if teacher_force:
                action = teacher_action(t, actions)
                reward_ = env.make_action(action)
            else:
                reward_, action = env.make_action_logits(logits)

            for i,a in enumerate(action):
                action_mask[i,t,a] = 1
            

            reward = [item[0] for item in reward_] # batch_size
            done = [item[1] for item in reward_]   # batch_size

            for i,e in enumerate(ended):
                if e:
                    reward[i] = 0

            
            rewards_list.append(reward)

            traj_length += (t+1) * np.logical_and(ended == 0, (done == 1))

            ended = np.logical_or(ended, (done == 1))

            if ended.all():
                break

        traj_length += seq_len * (ended == 0)
        for i,l in enumerate(traj_length):
            masks[i,:l] = 1

        for i in range(seq_len - len(state_list)):
            state_list.append(np.zeros_like(state_list[0]))
            finish_tag_list.append(np.zeros_like(finish_tag_list[0]))

        inputs_ = np.array(state_list).transpose(1, 0, 2, 3, 4) # bs x seq_len x H x W x C
        tags = np.array(finish_tag_list).transpose(1, 0, 2) # bs x seq_len x obj_num



        state, tag = env.get_state()

        logits = sess.run(net.output, feed_dict = {net.inputs_: np.expand_dims(state,1), net.finish_tag: np.expand_dims(tag,1), net.state_in:h_state})
        logits = logits.squeeze()
        
        last_value = logits.max(1) # bs
        
        discount_reward = np.zeros(batch_size, np.float32)

        for i in range(batch_size):
            if not ended[i]:
                discount_reward[i] = last_value[i]
        
        length = len(rewards_list)


        for t in range(length-1, -1, -1):
            discount_reward = rewards_list[t] + gamma * discount_reward

            Q_target[:,t] = discount_reward
            
        

        _, loss = sess.run([net_train.optimizer, net_train.loss], feed_dict={net_train.inputs_ : inputs_, net_train.target_Q_ : Q_target, net_train.finish_tag: tags, net_train.actions_: action_mask, net_train.lr: learning_rate, net_train.mask: masks})


        # summt.value.add(tag='learning rate',simple_value=lr)

        writer.add_scalar('loss',loss,step)
        print('iter',step,'loss', loss)
        # writer.add_summary(summt, int(step/optimize_frequency))

        def test_env(env,iters=None):
            bs = env.batch_size
            z_state_t = np.zeros([bs,dim_h])
            init_state_t = (z_state_t,z_state_t)

            if iters is None:
                iters = int(len(env.data) / env.batch_size)+1
            
            success_rate = []
            average_step = []
            average_reward = []
            for _ in range(iters):
                env.reset()
                h_state = init_state_t

                ended = np.array([False] * bs)
                traj_length = np.zeros(bs).astype(np.int32)
                reward_sum = np.zeros(bs).astype(np.float32)
                for t in range(seq_len):
                    state, tag = env.get_state()

                    logits, h_state = sess.run([net.output,net.state_out], feed_dict = {net.inputs_: np.expand_dims(state,1), net.finish_tag: np.expand_dims(tag,1), net.state_in:h_state})

                    logits = logits.squeeze()

                    rewards_, action = env.make_action_logits(logits)

                    rewards = np.array([item[0] for item in rewards_])
                    done = np.array([item[1] for item in rewards_])

                    traj_length += (t+1) * np.logical_and(ended == 0, (done == 1))
            
                    ended = np.logical_or(ended, (done == 1))
                    
                    for i,e in enumerate(ended):
                        if e:
                            rewards[i] = 0
                    
                    reward_sum += rewards

                    if ended.all():
                        break

                traj_length += seq_len * (ended == 0)
                average_reward.append(reward_sum)
                success_rate.append(ended.sum()/bs)
                average_step.append(traj_length.mean())
            
            return np.mean(success_rate), np.mean(average_step), np.mean(average_reward)
            

        if step % 100 == 0 and step > 0:
            sr,ms,mr = test_env(env_test_train, 5)

            writer.add_scalar('train sr',sr,step)
            writer.add_scalar('train step',ms,step)
            writer.add_scalar('train rewards',mr,step)

            print('train','sr',sr,'step',ms,'rewards',mr)
            for i,env_test in enumerate(env_tests):
                sr,ms,mr = test_env(env_test)

                writer.add_scalar('sr_%d'%(i*5+5),sr,step)
                writer.add_scalar('step_%d'%(i*5+5),ms,step)
                writer.add_scalar('rewards_%d'%(i*5+5),mr,step)            
                print('test',(i*5+5),'sr',sr,'step',ms,'rewards',mr)
                

        if step % 100 == 0 and step > 0: # !!!!! have been modified!!
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))



if __name__ == '__main__':
    train_net18_2_onehotchannel()
