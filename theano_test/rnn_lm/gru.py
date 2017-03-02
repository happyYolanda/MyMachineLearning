
# coding: utf-8

# In[ ]:

import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from collections import OrderedDict
import time
import operator

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def uniform_weight(shape):
    return np.random.uniform(-np.sqrt(1./shape[-1]), np.sqrt(1./shape[-1]), shape)

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzipp(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

class GRUTheano:
    
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #inputs
        self.x = T.ivector('x')
        self.y = T.ivector('y')
        # Initialize the network parameters
        params = {}
        params["E"] = uniform_weight((word_dim, hidden_dim)).astype(theano.config.floatX)
        params["U"] = uniform_weight((6, hidden_dim, hidden_dim)).astype(theano.config.floatX)
        params["W"] = uniform_weight((6, hidden_dim, hidden_dim)).astype(theano.config.floatX)
        params["V"] = uniform_weight((hidden_dim, word_dim)).astype(theano.config.floatX)
        params["b"] = np.zeros((6, hidden_dim)).astype(theano.config.floatX)
        params["c"] = np.zeros(word_dim).astype(theano.config.floatX)
        self.params = init_tparams(params)
        # We store the Theano graph here
        self.theano = {}
        self.__build__()
    
    def __build__(self):
        
        x = self.x
        y = self.y
        
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            
            # Word embedding layer
            x_e = self.params["E"][x_t,:]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(x_e.dot(self.params["U"][0]) + s_t1_prev.dot(self.params["W"][0]) + self.params["b"][0])
            r_t1 = T.nnet.hard_sigmoid(x_e.dot(self.params["U"][1]) + s_t1_prev.dot(self.params["W"][1]) + self.params["b"][1])
            c_t1 = T.tanh(x_e.dot(self.params["U"][2]) + (s_t1_prev * r_t1).dot(self.params["W"][2]) + self.params["b"][2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(s_t1.dot(self.params["U"][3]) + s_t2_prev.dot(self.params["W"][3]) + self.params["b"][3])
            r_t2 = T.nnet.hard_sigmoid(s_t1.dot(self.params["U"][4]) + s_t2_prev.dot(self.params["W"][4]) + self.params["b"][4])
            c_t2 = T.tanh(s_t1.dot(self.params["U"][5]) + (s_t2_prev * r_t2).dot(self.params["W"][5]) + self.params["b"][5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(s_t2.dot(self.params["V"]) + self.params["c"])[0]

            return [o_t, s_t1, s_t2]
        
        [output, s1, s2], updates = theano.scan(
            forward_prop_step,
            sequences = x,
            truncate_gradient = self.bptt_truncate,
            outputs_info=[None, dict(initial = T.zeros(self.hidden_dim)), dict(initial = T.zeros(self.hidden_dim))])
        
        self.cost = T.sum(T.nnet.categorical_crossentropy(output, y))
        
        # Gradients
        print "compile grads..."
        self.grads = T.grad(self.cost, [vv for kk, vv in self.params.iteritems()])
        
        # Assign functions
        self.predict = theano.function([x], output)
        self.predict_class = theano.function([x], T.argmax(output, axis=1))
        self.cost_out = theano.function([x, y], self.cost)    
        
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return np.sum([self.cost_out(x, y) for x, y in zip(X, Y)])/ float(num_words)
    
    def load_params(self, path):
        print ("loading model from %s..." % path)
        pp = np.load(path)
        params = unzipp(self.params)
        for kk, vv in params.iteritems():
            if kk not in pp:
                warnings.warn('%s is not in the model file' % kk)
                continue
            params[kk] = pp[kk]
        zipp(params, self.params) 
        
    def save_params(self, path):
        np.savez(path, **unzipp(self.params))
        