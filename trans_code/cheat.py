import theano
import theano.tensor as T


def theano_neg_log_likelihood():
    """
    Cheating
    """
    y = T.ivector('y')
    x = T.dmatrix('x')
    w = T.dmatrix('w')
    b = T.dvector('b')
    input = T.dot(x, w) + b
    p_y_given_x = T.nnet.softmax(input)

    return x, w, b, y, \
        -T.mean(
            T.log(
                p_y_given_x)[
                    T.arange(y.shape[0]), y])


def theano_neg_log_likelihood_prime_w():
    x, w, b, y, neg_like = theano_neg_log_likelihood()

    g = T.grad(cost=neg_like, wrt=w)

    return theano.function([x, w, b, y], g)


def theano_neg_log_likelihood_prime_b():
    x, w, b, y, neg_like = theano_neg_log_likelihood()

    g = T.grad(cost=neg_like, wrt=b)

    return theano.function([x, w, b, y], g)

