import numpy
import theano
import theano.tensor as T
import config
import time


def theano_softmax():
    x = T.fmatrix('x')
    _y = T.nnet.softmax(x)
    f = theano.function([x], _y)
    return f



def theano_neq():
    a = T.fmatrix('a')
    b = T.fmatrix('b')

    y = T.neq(a, b)

    f = theano.function([a, b], y)

    return f


def theano_p_y_given_x():
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.dvector('b')

    input = T.dot(x, w) + b
    y = T.nnet.softmax(input)

    f = theano.function([x, w, b], y)

    return f


def theano_argmax():
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.dvector('b')

    input = T.dot(x, w) + b

    y = T.nnet.softmax(input)
    a = T.argmax(y, axis=1)

    f = theano.function([x, w, b], a)

    return f


def theano_neg_log_likelihood():
    y = T.ivector('y')
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.dvector('b')
    input = T.dot(x, w) + b
    p_y_given_x = T.nnet.softmax(input)

    neg_like =\
        -T.mean(
            T.log(
                p_y_given_x)[
                    T.arange(y.shape[0]), y])

    return theano.function([x, w, b, y], neg_like)


def theano_neg_log_likelihood_prime():
    y = T.ivector('y')
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.dvector('b')
    input = T.dot(x, w) + b
    p_y_given_x = T.nnet.softmax(input)

    neg_like =\
        -T.mean(
            T.log(
                p_y_given_x)[
                    T.arange(y.shape[0]), y])

    g_W = T.grad(cost=neg_like, wrt=w)
    return theano.function([x, w, b, y], g_W)


def neq(a, b):
    return numpy.invert(numpy.equal(a, b)).astype('int32')


def softmax(w):
    w = numpy.array(w)

    #print "w = ", w

    maxes = numpy.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = numpy.exp(w - maxes)

    #print "e =", e

    dist = e / numpy.sum(e, axis=1).reshape(maxes.shape[0], 1)
    return dist


def p_y_given_x(X, w, b):
    dt = numpy.dot(X, w) + b
    return softmax(dt)


def argmax(X, w, b):
    return numpy.argmax(p_y_given_x(X, w, b), axis=1)


def neg_log_likelihood(X, w, b, y):
    r = numpy.arange(y.shape[0])
    l = numpy.log(p_y_given_x(X, w, b))

    return -numpy.mean(
        numpy.log(
            p_y_given_x(X, w, b))[
                numpy.arange(y.shape[0]), y])

theano_time = 0
our_time = 0

theano_procs = {
    'neq': theano_neq(),
    'softmax': theano_softmax(),
    'argmax': theano_argmax(),
    'p_y_given_x': theano_p_y_given_x(),
    'neg_log_likelihood': theano_neg_log_likelihood()
}

#num_samples = 10
#num_features = 10
#num_outputs = 10

num_samples = 50000
num_features = 784
num_outputs = 10

for i in range(1):
    
    # floatX = float32 in config
    X = numpy.array(numpy.random.rand(num_samples, num_features), dtype=config.floatX)
    X2 = numpy.array(numpy.random.rand(num_samples, num_features), dtype=config.floatX)
    w = numpy.array(numpy.random.rand(num_features, num_outputs), dtype=config.floatX)
    b = numpy.array(numpy.random.rand(num_outputs), dtype=config.floatX)
    y = numpy.array(numpy.random.random_integers(0, num_outputs-1, num_samples), dtype='int32')

    start_time = time.time()
    theirs = theano_procs['neq'](X, X2)
    theirs2 = theano_procs['neq'](X, X)
    theano_time += (time.time() - start_time)

    start_time = time.time()
    ours = neq(X, X2)
    ours2 = neq(X, X)
    our_time += (time.time() - start_time)

    #print "---------------------"
    #print "Theano"
    #print theirs
    #print theirs2
    #print "Ours"
    #print ours
    #print ours2
    #print "---------------------"
    #print ""

    assert numpy.array_equal(theirs, ours)
    assert numpy.array_equal(theirs2, ours2)

    start_time = time.time()
    theirs = theano_procs['softmax'](X)
    theano_time += (time.time() - start_time)

    start_time = time.time()
    ours = softmax(X)
    our_time += (time.time() - start_time)

    #print "---------------------"
    #print "Theano"
    #print theirs
    #print "Ours"
    #print ours
    #print "---------------------"
    #print ""


    assert numpy.allclose(theirs, ours, 0.005)

    start_time = time.time()
    theirs = theano_procs['p_y_given_x'](X, w, b)
    theano_time += (time.time() - start_time)

    start_time = time.time()
    ours = p_y_given_x(X, w, b)
    our_time += (time.time() - start_time)

    #print "---------------------"
    #print "Theano P(y) given X:"
    #print theirs
    #print "Our P(y) given X:"
    #print ours
    #print "---------------------"
    #print ""

    assert numpy.allclose(theirs, ours, 0.005)

    start_time = time.time()
    theirs = theano_procs['argmax'](X, w, b)
    theano_time += (time.time() - start_time)

    start_time = time.time()
    ours = argmax(X, w, b)
    our_time += (time.time() - start_time)

    #print "---------------------"
    #print "Theano argmax:"
    #print theirs
    #print "Our argmax"
    #print ours
    #print "---------------------"
    #print ""

    assert numpy.allclose(theirs, ours, 0.005)

    start_time = time.time()
    theirs = theano_procs['neg_log_likelihood'](X, w, b, y)
    theano_time += (time.time() - start_time)

    start_time = time.time()
    ours = neg_log_likelihood(X, w, b, y)
    our_time += (time.time() - start_time)

    #print "---------------------"
    #print "Theano negative log likelihood:"
    #print theirs
    #print "Our negative log likelihood"
    #print ours
    #print "---------------------"
    #print ""

    assert numpy.allclose(theirs, ours, 0.005)


print "Theano Time", theano_time
print "Our time", our_time
