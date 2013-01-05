"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import numpy
import nnet
import cheat
import matrix

import config


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = numpy.zeros((n_in, n_out), dtype=config.floatX)

        # initialize the baises b as a vector of n_out 0s
        self.b = numpy.zeros((n_out,), dtype=config.floatX)

        # parameters of the model
        self.params = [self.W, self.b]

    def p_y_given_x(self, X):
        # compute vector of class-membership probabilities in symbolic form
        dt = numpy.dot(X, self.W) + self.b
        return nnet.softmax(dt)

    def y_pred(self, X):
        # compute prediction as class whose probability is maximal in
        # symbolic form
        return numpy.argmax(self.p_y_given_x(X), axis=1)


    def negative_log_likelihood(self, X, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.

        p_y = self.p_y_given_x(X)

        return\
            -numpy.mean(
                numpy.log(
                    p_y)[
                        numpy.arange(y.shape[0]), y])

    def errors(self, X, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return numpy.mean(matrix.neq(self.y_pred(X), y))


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    test_set_y = numpy.array(test_set_y, dtype='int32')
    valid_set_y = numpy.array(valid_set_y, dtype='int32')
    train_set_y = numpy.array(train_set_y, dtype='int32')

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

## Cheat and use theano for automatic differentiation
g_w_fn = cheat.theano_neg_log_likelihood_prime_w()
g_b_fn = cheat.theano_neg_log_likelihood_prime_b()


def train_model(
        classifier, learning_rate, train_set_x, train_set_y, minibatch_index, batch_size):

    X = train_set_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
    y = train_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

    ## the cost we minimize during training is the negative log likelihood of
    ## the model in symbolic format
    cost = classifier.negative_log_likelihood(X, y)

    ## compute the gradient of cost with respect to theta = (W,b)
    g_W = g_w_fn(X, classifier.W, classifier.b, y)
    g_b = g_b_fn(X, classifier.W, classifier.b, y)

    ## specify how to update the parameters of the model as a dictionary
    classifier.W = classifier.W - learning_rate * g_W
    classifier.b = classifier.b - learning_rate * g_b


def test_model(
        classifier, valid_set_x, valid_set_y, minibatch_index, batch_size):
    X = valid_set_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
    y = valid_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

    return classifier.errors(X, y)


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='../data/mnist.pkl.gz',
                           batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(n_in=28 * 28, n_out=10)

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            # Note: This doesn't actually return the average cost, bug in the
            # tutorial
            train_model(
                classifier, learning_rate, train_set_x, train_set_y,
                minibatch_index, batch_size)
            ## iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ## compute zero-one loss on validation set
                validation_losses = [test_model(
                    classifier, valid_set_x, valid_set_y, i,
                    batch_size) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                ## if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    ## test it on the test set

                    test_losses = [test_model(
                        classifier, test_set_x, test_set_y, i,
                        batch_size) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        ('     epoch %i, minibatch %i/%i, test error of best'
                         ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
          (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    sgd_optimization_mnist()
