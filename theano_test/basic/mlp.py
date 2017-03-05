import numpy

import theano
import theano.tensor as T
from LogisticRegression import *

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh, level = '1'):
        
        self.input = input
       
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        W = theano.shared(value=W_values, name=('W_{}'.format(level)), borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name=('b_{}'.format(level)), borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh,
            level='1'
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,      #last layer's output is this layer's input
            n_in=n_hidden,
            n_out=n_out
        ) 
        # L1 norm 
        self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())

        # square of L2 norm 
        self.L2 = ((self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())

        self.loss = self.logRegressionLayer.loss
        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # keep track of model input
        self.input = input
        self.y_pred = self.logRegressionLayer.y_pred
    
    def save(self, filename):
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)
        
    def load(self, filename):
        vals = numpy.load(filename)
        for p in self.params:
            if p.name in vals:
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception("Shape mismatch: {} != {} for {}"
                            .format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(vals[p.name])
            else:
                raise Exception("model wrong!!! need param {} ".format(p.name))
