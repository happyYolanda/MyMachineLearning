import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
 
    def __init__(self, input, n_in, n_out):
        #initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),  
            name='W',
            borrow=True
        )   
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),  
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
        
    def loss(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
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
  

