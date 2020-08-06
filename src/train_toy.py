import tensorflow as tf
import numpy as np
import os
from env import ENV
from env import ENV3
from collections import deque


class Toy:
    def __init__(self, state_size=[5,5], action_size=50, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            

            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            

        
            
            output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

            self.output = tf.nn.softmax(output)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.actions_))
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            # self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Toy2:
    def __init__(self, state_size=[5,5,50], action_size=50, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            

            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            

        
            
            output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

            self.output = tf.nn.softmax(output)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.actions_))
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            # self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size=3000000):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(buffer_size,
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

def data(batch_size):
    x = []
    y = []
    size = [5,5]
    for i in range(batch_size):
        m = np.zeros(size)
        _x = np.random.randint(5)
        _y = np.random.randint(5)
        _x_ = np.random.randint(_x, 5)
        _y_ = np.random.randint(_y, 5)
        l = np.random.randint(50) + 1
        m[_x:_x_+1, _y:_y_+1] = l
        label = np.zeros(50)
        label[l-1] = 1
        x.append(m)
        y.append(label)

    return x,y


def data2(batch_size):
    x = []
    y = []
    size = [5,5,50]
    for i in range(batch_size):
        m = np.zeros(size)
        _x = np.random.randint(5)
        _y = np.random.randint(5)
        _x_ = np.random.randint(_x, 5)
        _y_ = np.random.randint(_y, 5)
        l = np.random.randint(50)
        m[_x:_x_+1, _y:_y_+1, l] = 1
        label = np.zeros(50)
        label[l] = 1
        x.append(m)
        y.append(label)

    return x,y



def train():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 3
    start_to_train = 100000
    state_size = [15,15]


    tensorboard_path = "tensorboard/toy/"
    weight_path = "toy_weights"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = Toy()
    
    '''
        Setup buffer
    '''
    buffer = Memory()


    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    
    sess = tf.InteractiveSession()
    
    sess.run(tf.global_variables_initializer())
    decay_step = 0

    
    for step in range(total_episodes):
        x, y = data(batch_size) 
        

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : x, net.actions_: y})
        
        writer.add_summary(summary,step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))



def train2():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 3


    tensorboard_path = "tensorboard/toy2000/"
    weight_path = "toy2000_weights"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = Toy2(learning_rate=learning_rate)
    
    '''
        Setup buffer
    '''
    buffer = Memory()


    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    
    sess = tf.InteractiveSession()
    
    sess.run(tf.global_variables_initializer())
    decay_step = 0

    
    for step in range(total_episodes):
        x, y = data2(batch_size) 
        

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : x, net.actions_: y})
        
        writer.add_summary(summary,step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def test1():
    net = Toy()

    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())
    
    # saver.restore(sess,'toy_weights/model_440000.ckpt')
    saver.restore(sess,'toy_weights/model_70000.ckpt')
    
    x,y = data(60)
    
    out = sess.run(net.output,feed_dict={net.inputs_: x})
    for i,_ in enumerate(y):
        _x = x[i]
        for __ in _x:
            print('[%.0f %.0f %.0f %.0f %.0f]'%(__[0],__[1],__[2],__[3],__[4]))
        # print('[%.2f %.2f %.2f %.2f %.2f]'%(out[i,0],out[i,1],out[i,2],out[i,3],out[i,4]),_)
        print(np.argmax(out[i]),np.argmax(_))


if __name__ == '__main__':
    # train()
    train2()
    # test1()
    # test3()
    # from visulization import convert
    # import numpy as np
    # from PIL import Image
    # m = np.zeros([5,5])
    # m[0,0] = 4
    # img = convert(m)
    # img = Image.fromarray(np.uint8(img))
    # img.save('what.png')