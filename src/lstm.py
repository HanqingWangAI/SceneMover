import tensorflow as tf
class DQNetwork16:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            #mask
            self.mask = tf.placeholder(tf.float32, [None, self.seq_len])

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 128,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 256,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 512,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            
                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
    
                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, init_state = lstm_layer(256, 2, batch_size)
            outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])

            #self.fc1 = tf.layers.dense(inputs = self.rnn,
            #                      units = 256,
            #                      activation = tf.nn.elu,
            #                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                    name="fc1")

            # self.fc2 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")

            print(self.output_)
            print(self.output)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            print("actions_")
            print(self.actions_.shape)
            self.actions = tf.reshape(self.actions_, [-1, self.action_size])
            self.Q = tf.multiply(self.output_, self.actions, name = "Q")
            print(self.Q)
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.target_Q = tf.reshape(self.target_Q_, [-1, self.action_size])
            temp = tf.square(self.target_Q - self.Q)

            loss_details = tf.reduce_mean(tf.reshape(temp,[-1, num_objects, action_space]),axis=[0,1], name = "loss_details")
            print(loss_details)
            
            self.loss_details = [loss_details[i] for i in range(action_space)]
            
            temp = tf.reshape(tf.reduce_mean(temp, axis = 1), [-1, seq_len])
            self.loss = tf.reduce_mean(tf.multiply(temp, self.mask))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork16_eval:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name, reuse = True):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)
            #self.state_in = ((tf.placeholder(tf.float32, [None, 256], name = "state_in_c1"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h1")),
            #                (tf.placeholder(tf.float32, [None, 256], name = "state_in_c2"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h2")))
            self.state_in = (tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c1"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h1")),
                                 tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c2"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h2")))
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 128,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 256,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 512,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            #print("finish_tag")
            #print(self.finish_tag_.shape)
            self.flatten_ = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            #print("flatten_")
            #print(self.flatten_.shape)
            self.flatten = tf.reshape(self.flatten_, [-1, self.seq_len, int(self.flatten_.shape[-1])])
            #print("flatten")
            #print(self.flatten.shape)
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            
                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, self.init_state = lstm_layer(256, 2)
            self.rnn, self.state_out = tf.nn.dynamic_rnn(cell, self.flatten, initial_state = self.state_in)
            print(self.rnn)
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")
            
            print(self.output_)
            print(self.output)


class DQNetwork17:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            #mask
            self.mask = tf.placeholder(tf.float32, [None, self.seq_len])
            self.lr = tf.placeholder(tf.float32, name="learnig_rate")

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            
                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
    
                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, init_state = lstm_layer(256, 2, batch_size)
            outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])

            #self.fc1 = tf.layers.dense(inputs = self.rnn,
            #                      units = 256,
            #                      activation = tf.nn.elu,
            #                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                    name="fc1")

            # self.fc2 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")

            print(self.output_)
            print(self.output)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            print("actions_")
            print(self.actions_.shape)
            self.actions = tf.reshape(self.actions_, [-1, self.action_size])
            self.Q = tf.multiply(self.output_, self.actions, name = "Q")
            print(self.Q)
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.target_Q = tf.reshape(self.target_Q_, [-1, self.action_size])
            temp = tf.square(self.target_Q - self.Q)

            loss_details = tf.reduce_mean(tf.reshape(temp,[-1, num_objects, action_space]),axis=[0,1], name = "loss_details")
            print(loss_details)
            
            self.loss_details = [loss_details[i] for i in range(action_space)]
            
            temp = tf.reshape(tf.reduce_mean(temp, axis = 1), [-1, seq_len])
            self.loss = tf.reduce_mean(tf.multiply(temp, self.mask))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork17_eval:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name, reuse = True):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)
            #self.state_in = ((tf.placeholder(tf.float32, [None, 256], name = "state_in_c1"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h1")),
            #                (tf.placeholder(tf.float32, [None, 256], name = "state_in_c2"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h2")))
            self.state_in = (tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c1"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h1")),
                                 tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c2"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h2")))
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            #print("finish_tag")
            #print(self.finish_tag_.shape)
            self.flatten_ = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            #print("flatten_")
            #print(self.flatten_.shape)
            self.flatten = tf.reshape(self.flatten_, [-1, self.seq_len, int(self.flatten_.shape[-1])])
            #print("flatten")
            #print(self.flatten.shape)
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)
            
                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, self.init_state = lstm_layer(256, 2)
            self.rnn, self.state_out = tf.nn.dynamic_rnn(cell, self.flatten, initial_state = self.state_in)
            print(self.rnn)
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")
            
            print(self.output_)
            print(self.output)



class DQNetwork18:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            #mask
            
            self.reward_mask = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size])
            self.lr = tf.placeholder(tf.float32, name="learnig_rate")

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell_f(size):
                    return tf.nn.rnn_cell.LSTMCell(size, name='basic_lstm_cell')
            
                # cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
                cell = cell_f(lstm_size)

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, init_state = lstm_layer(256, 1, batch_size)
            outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])


            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")

            print(self.output_)
            print(self.output)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            print("actions_")
            # print(self.actions_.shape)
            self.actions = tf.reshape(self.actions_, [-1, self.action_size])
            # self.Q = tf.multiply(self.output_, self.actions, name = "Q")
            # print(self.Q)
            self.Q = self.output_

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.target_Q = tf.reshape(self.target_Q_, [-1, self.action_size])
            temp = tf.square(self.target_Q - self.Q)
            temp = tf.multiply(tf.reshape(temp, [-1, self.seq_len, self.action_size]),self.reward_mask)

            loss_details = tf.reduce_mean(tf.reshape(temp,[-1, num_objects, action_space]),axis=[0,1], name = "loss_details")
            print(loss_details)
            
            self.loss_details = [loss_details[i] for i in range(action_space)]
            
            # temp = tf.reshape(tf.reduce_mean(temp, axis = 1), [-1, seq_len])
            # self.loss = tf.reduce_mean(tf.multiply(temp, self.mask))
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            # self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork18_eval:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name, reuse = True):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)
            #self.state_in = ((tf.placeholder(tf.float32, [None, 256], name = "state_in_c1"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h1")),
            #                (tf.placeholder(tf.float32, [None, 256], name = "state_in_c2"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h2")))
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c1"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h1"))
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)


            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            #print("finish_tag")
            #print(self.finish_tag_.shape)
            self.flatten_ = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            #print("flatten_")
            #print(self.flatten_.shape)
            self.flatten = tf.reshape(self.flatten_, [-1, self.seq_len, int(self.flatten_.shape[-1])])
            #print("flatten")
            #print(self.flatten.shape)
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell_f(size):
                    return tf.nn.rnn_cell.LSTMCell(lstm_size, name='basic_lstm_cell')
            
                # cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
                cell = cell_f(lstm_size)

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, self.init_state = lstm_layer(256, 1)
            self.rnn, self.state_out = tf.nn.dynamic_rnn(cell, self.flatten, initial_state = self.state_in)
            print(self.rnn)
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")
            
            print(self.output_)
            print(self.output)


class DQNetwork18_2:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q_ = tf.placeholder(tf.float32, [None, self.seq_len], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            #mask
            
            self.mask = tf.placeholder(tf.float32, [None, self.seq_len])
            self.lr = tf.placeholder(tf.float32, name="learnig_rate")

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell_f(size):
                    return tf.nn.rnn_cell.LSTMCell(size, name='basic_lstm_cell')
            
                # cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
                cell = cell_f(lstm_size)

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, init_state = lstm_layer(256, 1, batch_size)
            outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])


            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")

            print(self.output_)
            print(self.output)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=2) # bs x seq_len

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.target_Q = self.target_Q_
            temp = tf.square(self.target_Q - self.Q) # bs x seq_len
            temp = tf.multiply(temp, self.mask)

            # loss_details = tf.reduce_mean(tf.reshape(temp,[-1, num_objects, action_space]),axis=[0,1], name = "loss_details")
            # print(loss_details)
            
            # self.loss_details = [loss_details[i] for i in range(action_space)]
            
            # temp = tf.reshape(tf.reduce_mean(temp, axis = 1), [-1, seq_len])
            # self.loss = tf.reduce_mean(tf.multiply(temp, self.mask))
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            # self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)


class DQNetwork19:
    # with tag only
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.seq_len, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q_ = tf.placeholder(tf.float32, [None, self.seq_len], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            #mask
            
            self.mask = tf.placeholder(tf.float32, [None, self.seq_len])
            self.lr = tf.placeholder(tf.float32, name="learnig_rate")

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            # self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            # self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell_f(size):
                    return tf.nn.rnn_cell.LSTMCell(size, name='basic_lstm_cell')
            
                # cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
                cell = cell_f(lstm_size)

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            # cell, init_state = lstm_layer(256, 1, batch_size)
            # outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            outputs = tf.layers.dense(inputs = self.flatten, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = 256, 
                                           activation=tf.nn.elu,
                                           name = "dense")

            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])


            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")

            print(self.output_)
            print(self.output)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=2) # bs x seq_len

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.target_Q = self.target_Q_
            temp = tf.square(self.target_Q - self.Q) # bs x seq_len
            temp = tf.multiply(temp, self.mask)

            # loss_details = tf.reduce_mean(tf.reshape(temp,[-1, num_objects, action_space]),axis=[0,1], name = "loss_details")
            # print(loss_details)
            
            # self.loss_details = [loss_details[i] for i in range(action_space)]
            
            # temp = tf.reshape(tf.reduce_mean(temp, axis = 1), [-1, seq_len])
            # self.loss = tf.reduce_mean(tf.multiply(temp, self.mask))
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            # self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)


class DQNetwork19_eval:
    def __init__(self, batch_size, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, seq_len = 50, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        
        with tf.variable_scope(name, reuse = True):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, self.seq_len, num_objects], name="finish_tag")
            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)
            #self.state_in = ((tf.placeholder(tf.float32, [None, 256], name = "state_in_c1"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h1")),
            #                (tf.placeholder(tf.float32, [None, 256], name = "state_in_c2"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h2")))
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name = "lstm_c1"),
                                                               tf.placeholder(tf.float32, [None, 256], name = "lstm_h1"))
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])# combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 64,
                                         kernel_size = [5,5],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 64,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 128,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 128,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 256,
                                         kernel_size = [3,3],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 256,
                                 kernel_size = [1,1],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = False,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)


            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            #print("finish_tag")
            #print(self.finish_tag_.shape)
            self.flatten_ = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            #print("flatten_")
            #print(self.flatten_.shape)
            # self.flatten = tf.reshape(self.flatten_, [-1, self.seq_len, int(self.flatten_.shape[-1])])
            #print("flatten")
            #print(self.flatten.shape)
            ## --> [1152]
            
            def lstm_layer(lstm_size, number_of_layers):
                '''
                This method is used to create LSTM layer/s for PixelRNN
    
                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network
              
                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''
                def cell_f(size):
                    return tf.nn.rnn_cell.LSTMCell(lstm_size, name='basic_lstm_cell')
            
                # cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])
                cell = cell_f(lstm_size)

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            # cell, self.init_state = lstm_layer(256, 1)
            # self.rnn, self.state_out = tf.nn.dynamic_rnn(cell, self.flatten, initial_state = self.state_in)
            # print(self.rnn)
            self.rnn = tf.layers.dense(inputs = self.flatten_, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = 256, 
                                           activation=tf.nn.elu,
                                           name = "dense")
            
            self.output_ = tf.layers.dense(inputs = self.rnn, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units = self.action_size, 
                                           activation=None,
                                           name = "output_internal")
            self.output = tf.reshape(self.output_, [-1, self.seq_len, self.action_size], name = "output_external")
            
            print(self.output_)
            print(self.output)
