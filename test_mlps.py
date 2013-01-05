import numpy
import random
import theano
import theano.tensor as T
from code import HiddenLayer as theano_HiddenLayer, MLP as theano_MLP
from trans_code import HiddenLayer, MLP, train_model, test_model, load_data

rng = numpy.random.RandomState(1234)

num_samples = 1000
num_hidden = 3
num_features = 5
num_outputs = 3

sx = T.matrix('x')
sy = T.ivector('y')


def theano_train_model(tmlp, learning_rate, L1_reg, L2_reg, sx, sy):
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = tmlp.negative_log_likelihood(sy)\
        + L1_reg * tmlp.L1 \
        + L2_reg * tmlp.L2_sqr

    #validate_model = theano.function(inputs=[index],
                                     #outputs=classifier.errors(y),
                                     #givens={
                                     #x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                     #y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in tmlp.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a dictionary
    updates = {}
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(tmlp.params, gparams):
        updates[param] = param - learning_rate * gparam

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[sx, sy], outputs=cost,
                                  updates=updates)

    ## compiling a Theano function that computes the mistakes that are made
    ## by the model on a minibatch
    test_model = theano.function(inputs=[sx, sy],
                                 outputs=tmlp.errors(sy))

    return train_model, test_model


def run_test():
    learning_rate = random.random()
    L1_reg = random.random()
    L2_reg = random.random()
    hl = HiddenLayer(
        rng, num_features, num_outputs, activation=numpy.tanh)

    thl = theano_HiddenLayer(
        rng, sx, num_features, num_outputs, W=hl.W, b=hl.b, activation=T.tanh)

    mlp = MLP(
        rng, n_in=num_features, n_hidden=num_hidden, n_out=num_outputs)

    # Ensure that the theano MLP shares the same weights & biases as our MLP
    hW = theano.shared(value=numpy.copy(mlp.hiddenLayer.W), name='W', borrow=True)
    hb = theano.shared(value=numpy.copy(mlp.hiddenLayer.b), name='b', borrow=True)

    tmlp = theano_MLP(
        rng, sx, n_in=num_features, n_hidden=num_hidden, n_out=num_outputs, hW=hW, hb=hb)

    thl_out = theano.function([sx], thl.output)
    tmlp_p_y_given_x = theano.function([sx], tmlp.logRegressionLayer.p_y_given_x)
    tmlp_neg_log = theano.function(
        inputs=[sx, sy],
        outputs=tmlp.negative_log_likelihood(sy))

    tmlp_train_model, tmlp_test_model = theano_train_model(tmlp, learning_rate, L1_reg, L2_reg, sx, sy)

    X = numpy.array(numpy.random.rand(num_samples, num_features), dtype=theano.config.floatX)
    y = numpy.array(numpy.random.random_integers(0, num_outputs-1, num_samples), dtype='int32')

    theirs = thl_out(X)
    ours = hl.output(X)
    assert numpy.allclose(theirs, ours, 0.005)

    theirs = tmlp_p_y_given_x(X)
    ours = mlp.logRegressionLayer.p_y_given_x(mlp.hiddenLayer.output(X))
    assert numpy.allclose(theirs, ours, 0.0000001)

    theirs = tmlp_neg_log(X, y)
    ours = mlp.negative_log_likelihood(X, y)
    assert numpy.allclose(theirs, ours, 0.0000001)

    for j in range(10):
        X = numpy.array(numpy.random.rand(num_samples, num_features), dtype=theano.config.floatX)
        y = numpy.array(numpy.random.random_integers(0, num_outputs-1, num_samples), dtype='int32')

        test_X = numpy.array(numpy.random.rand(num_samples, num_features), dtype=theano.config.floatX)
        test_y = numpy.array(numpy.random.random_integers(0, num_outputs-1, num_samples), dtype='int32')

        train_model(mlp, learning_rate, X, y, 0, num_samples, L1_reg, L2_reg)
        tmlp_train_model(X, y)

        print "Testing regression layer..."
        theirs = tmlp.logRegressionLayer.W.get_value(borrow=True)
        ours = mlp.logRegressionLayer.W
        assert numpy.allclose(theirs, ours, 0.005)

        theirs = tmlp.logRegressionLayer.b.get_value(borrow=True)
        ours = mlp.logRegressionLayer.b
        assert numpy.allclose(theirs, ours, 0.005)

        print "Testing hidden layer..."
        theirs = tmlp.hiddenLayer.W.get_value(borrow=True)
        ours = mlp.hiddenLayer.W
        assert numpy.allclose(theirs, ours, 0.005)

        theirs = tmlp.hiddenLayer.b.get_value(borrow=True)
        ours = mlp.hiddenLayer.b
        assert numpy.allclose(theirs, ours, 0.005)

        theirs = tmlp_test_model(test_X, test_y)
        ours = test_model(mlp, test_X, test_y, 0, num_samples)
        assert numpy.allclose(theirs, ours, 0.005)


def do_mnist():
    num_features = 784
    num_hidden = 500
    num_outputs = 10
    L1_reg = 0.00
    L2_reg = 0.0001
    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 500

    datasets = load_data('data/mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    mlp = MLP(
        rng, n_in=num_features, n_hidden=num_hidden, n_out=num_outputs)

    ## Ensure that the theano MLP shares the same weights & biases as our MLP
    #hW = theano.shared(value=numpy.copy(mlp.hiddenLayer.W), name='W', borrow=True)
    #hb = theano.shared(value=numpy.copy(mlp.hiddenLayer.b), name='b', borrow=True)

    tmlp = theano_MLP(
        rng, sx, n_in=num_features, n_hidden=num_hidden, n_out=num_outputs)

    tmlp_train_model, tmlp_test_model = theano_train_model(tmlp, learning_rate, L1_reg, L2_reg, sx, sy)

    for epoch in range(num_epochs):
        print "Epoch", epoch
        for minibatch_index in xrange(n_train_batches):
            print "Minibatch", minibatch_index
            X = train_set_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

            train_model(mlp, learning_rate, X, y, 0, len(X), L1_reg, L2_reg)
            tmlp_train_model(X, y)

            #print "Testing regression layer..."
            #theirs = tmlp.logRegressionLayer.W.get_value(borrow=True)
            #ours = mlp.logRegressionLayer.W
            #print theirs
            #print ours
            #print format(numpy.sum(abs(theirs - ours)), 'f')
            #assert numpy.allclose(theirs, ours, 0.005)

            #theirs = tmlp.logRegressionLayer.b.get_value(borrow=True)
            #ours = mlp.logRegressionLayer.b
            #assert numpy.allclose(theirs, ours, 0.005)

            #print "Testing hidden layer..."
            #theirs = tmlp.hiddenLayer.W.get_value(borrow=True)
            #ours = mlp.hiddenLayer.W
            #assert numpy.allclose(theirs, ours, 0.005)

            #theirs = tmlp.hiddenLayer.b.get_value(borrow=True)
            #ours = mlp.hiddenLayer.b
            #assert numpy.allclose(theirs, ours, 0.005)

            print "Testing against test set..."
            theirs = tmlp_test_model(test_set_x, test_set_y)
            ours = test_model(
                mlp, test_set_x, test_set_y, 0, test_set_x.shape[0])

            print theirs
            print ours
            #assert numpy.allclose(theirs, ours, 0.005)

for i in range(2):
    run_test()

#do_mnist()
