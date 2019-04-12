__author__ = "Shreyas Seshadri, shreyas.seshadri@aalto.fi; Lauri Juvela, lauri.juvela@aalto.fi"

import os
import sys
import math
import numpy as np
import tensorflow as tf

_FLOATX = tf.float32 # todo: move to lib/precision.py

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
   

class CNET():

    def __init__(self,
                 name,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim = None,
                 cond_channels = 64,
                 postnet_channels=256,
                 do_postproc = True,
                 do_GLU = True):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GLU = do_GLU

        if cond_dim is not None:
            self._use_cond = True
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels
            
        else:
            self._use_cond = False

        self._name = name


    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)          


    def _batch_norm(self, x, phase_train, beta, gamma):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):
#            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                                         name='beta', trainable=True)
#            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
    
    
    def _instance_norm(self, x, beta, gamma):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('in'):
#            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                                         name='beta', trainable=True)
#            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [1], name='moments', keep_dims=True)

            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
        return normed
 
    def _input_layer(self, main_input):
        fw = self.filter_width
        r = self.residual_channels
        i = self.input_channels
        
        with tf.variable_scope('input_layer'):            

            W = get_weight_variable('W', (fw, i, 2*r))
            b = get_bias_variable('b', (2*r))

            X = main_input
            Y = tf.nn.convolution(X, W, padding='SAME')
            Y += b
            Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
            
        return Y

    def _embed_cond(self, cond_input):
        cd = self.cond_dim
        c = self.cond_channels
        with tf.variable_scope('embed_cond'):            
            
            W1 = get_weight_variable('W1',(1, cd, c))
            b1 = get_bias_variable('b1',(c))
            W2 = get_weight_variable('W2',(1, c, c))
            b2 = get_bias_variable('b2',(c))

            Y = tf.nn.convolution(cond_input, W1, padding='SAME') # 1x1 convolution
            Y += b1
            Y = tf.nn.leaky_relu(Y)
            
            Y = tf.nn.convolution(Y, W2, padding='SAME') # 1x1 convolution
            Y += b2
            Y = tf.nn.leaky_relu(Y)
            
            return Y


    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        fw = self.filter_width
        r = self.residual_channels
        
        s = self.postnet_channels
        
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):
                                
                W = get_weight_variable('filter_gate_W',(fw, r, 2*r)) 
                b = get_bias_variable('filter_gate_b',(2*r))                                                          

                X = main_input
                Y = tf.nn.convolution(X, W, padding='SAME', dilation_rate=[dilation])
                Y += b
                if self._use_cond:
                    c = self.cond_channels
                    beta_V = get_weight_variable('beta_V',(1, c, 2*r)) 
                    beta_b = get_bias_variable('beta_b',(2*r)) 
                    beta = tf.nn.convolution(cond_input, beta_V, padding='SAME') # 1x1 convolution
                    beta += beta_b
                    
                    gamma_V = get_weight_variable('gamma_V',(1, c, 2*r)) 
                    gamma_b = get_bias_variable('gamma_b',(2*r))
                    gamma = tf.nn.convolution(cond_input, gamma_V, padding='SAME') # 1x1 convolution
                    gamma += gamma_b
                    
                    Y = self._instance_norm(Y, beta, gamma)
                    #Y = tf.contrib.layers.instance_norm(X,param_initializers={'beta':beta,'gamma':gamma},trainable=False)
                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
                
                # add residual channel
                if self.do_postproc: 
                    W_s = get_weight_variable('skip_gate_W',(fw, r, s)) 
                    b_s = get_weight_variable('skip_gate_b',s)
                    
                    skip_out = tf.nn.convolution(Y, W_s, padding='SAME')
                    skip_out += b_s
                else:
                    skip_out = []
                
                if self.do_GLU:
                    W_p = get_weight_variable('post_filter_gate_W',(1, r, r)) 
                    b_p = get_weight_variable('post_filter_gate_b',r)
                    
                    Y = tf.nn.convolution(Y, W_p, padding='SAME')
                    Y += b_p
                    Y += X
                
                                          

        return Y, skip_out


    def _postproc_module(self, residual_module_outputs):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels
        o = self.output_channels
        
        with tf.variable_scope('postproc_module'):

            W1 = get_weight_variable('W1',(fw, r, s))
            b1 = get_bias_variable('b1',s)
            W2 = get_weight_variable('W2', (s, o))
            b2 = get_bias_variable('b2',o)

            # sum of residual module outputs
            X = tf.zeros_like(residual_module_outputs[0])
            for R in residual_module_outputs:
                X += R

            Y = tf.nn.convolution(X, W1, padding='SAME')    
            Y += b1
            Y = tf.nn.relu(Y)

            Y = tf.nn.convolution(Y, W2, padding='SAME')    
            Y += b2

            if type(self.output_channels) is list:
                #import ipdb; ipdb.set_trace()
                output_list = []
                start = 0 
                for channels in self.output_channels:
                    output_list.append(Y[:,:,start:start+channels])
                    start += channels
                Y = output_list
            
        return Y
    
    def _last_layer(self, last_layer_ip):
        fw = self.filter_width
        r = self.residual_channels
        o = self.output_channels
        
        with tf.variable_scope('last_layer'):

            W = get_weight_variable('W',(fw, r, o))
            b = get_bias_variable('b',o)


            X = last_layer_ip            

            Y = tf.nn.convolution(X, W, padding='SAME')    
            Y += b

            if type(self.output_channels) is list:
                #import ipdb; ipdb.set_trace()
                output_list = []
                start = 0 
                for channels in self.output_channels:
                    output_list.append(Y[:,:,start:start+channels])
                    start += channels
                Y = output_list
            
        return Y

    def forward_pass(self, X_input, cond_input=None):
        
        skip_outputs = []
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):

            if self._use_cond:
                #C = self._embed_cond(cond_input)
                C = cond_input
            else:
                C = None    

            R = self._input_layer(X_input)
            X = R
            for i, dilation in enumerate(self.dilations):
                X, skip = self._conv_module(X, R, i, dilation, cond_input=C)
                skip_outputs.append(skip)

            if self.do_postproc:    
                Y = self._postproc_module(skip_outputs)    
            else:
                Y = self._last_layer(X)                

        return Y
                                                 