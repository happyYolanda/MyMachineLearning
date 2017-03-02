import sys
import os
import numpy
import theano
import theano.tensor as TT
import gzip
import cPickle
import timeit
from LogisticRegression import *
from mlp import *

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset='data/mnist.pkl.gz'):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def train(lrate=0.01, n_epochs = 10, dataset = 'data/mnist.pkl.gz', batch_size = 20, L1_reg = 0.00, L2_reg = 0.0001, n_hidden = 500):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    index = T.lscalar()
    x = T.matrix("x")
    y = T.ivector("y")
    
    classifier = LogisticRegression(x, n_in = 28 * 28, n_out = 10)
    cost = classifier.loss(y)
    '''
    rng = numpy.random.RandomState(1234)
    classifier = MLP(rng=rng, input=x, n_in = 28 * 28, n_hidden = n_hidden, n_out = 10)
    cost = classifier.loss(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2
    '''
    grads = T.grad(cost, classifier.params)
    updates = [(p, p - lrate * g) for (p, g) in zip(classifier.params, grads)]
    
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print( 'epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('  epoch %i, minibatch %i/%i, test error of' ' best model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                    # save the best model
                    classifier.save("best_model.npz")
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%,' 'with test performance %f %%') % 
              (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +  ' ran for %.1fs' % ((end_time - start_time)))
    
def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    x = T.matrix("x")
    
    classifier = LogisticRegression(x, n_in = 28 * 28, n_out = 10)
    
    '''
    rng = numpy.random.RandomState(1234)
    classifier = MLP(rng=rng, input=x, n_in = 28 * 28, n_hidden = 500, n_out = 10)
    '''
    # load the saved model
    classifier.load('best_model.npz')
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='data/mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values
    print ("Real values for the first 10 examples in test set:")
    print test_set_y[:10].eval()
    
if __name__ == '__main__':
    train()
    predict()
