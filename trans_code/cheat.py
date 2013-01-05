import theano
import theano.tensor as T


def theano_neg_log_likelihood(x, w, b, y):
    """
    Cheating
    """

    input = T.dot(x, w) + b
    p_y_given_x = T.nnet.softmax(input)

    return \
        -T.mean(
            T.log(
                p_y_given_x)[
                    T.arange(y.shape[0]), y])


def theano_neg_log_likelihood_prime_w():
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.fvector('b')
    y = T.ivector('y')

    neg_like = theano_neg_log_likelihood(x, w, b, y)

    g = T.grad(cost=neg_like, wrt=w)

    return theano.function([x, w, b, y], g)


def theano_neg_log_likelihood_prime_b():
    x = T.fmatrix('x')
    w = T.fmatrix('w')
    b = T.fvector('b')
    y = T.ivector('y')

    neg_like = theano_neg_log_likelihood(x, w, b, y)

    g = T.grad(cost=neg_like, wrt=b)

    return theano.function([x, w, b, y], g)


def theano_mlp_prime():
    """
    Cheater function that uses theano to calculate the gradients of the negative
    log likelihood + L1/L2 regularization terms for the logistic and hidden
    layers of a multi layer perceptron
    """
    # Build all of the input symbols
    L1_reg_term = T.fscalar('L1_reg')
    L2_reg_term = T.fscalar('L2_reg')

    hidden_x = T.fmatrix('hidden_x')
    log_x = T.fmatrix('log_x')
    y = T.ivector('y')

    log_layer_w = T.fmatrix('w')
    log_layer_b = T.fvector('b')

    hidden_layer_w = T.fmatrix('hidden_w')
    hidden_layer_b = T.fvector('hidden_b')

    # Calculate the negative log likelihood of the hidden layer
    # on the input
    neg_like_hidden = theano_neg_log_likelihood(
        hidden_x, hidden_layer_w, hidden_layer_b, y)

    # Calculate the negative log likelihood of the logistic regresison
    # layer on the output of the hidden layer
    neg_like_log = theano_neg_log_likelihood(
        log_x, log_layer_w, log_layer_b, y)

    # L1 norm ; one regularization option is to enforce L1 norm to
    # be small
    L1_out = abs(hidden_layer_w).sum() + abs(log_layer_w).sum()

    # square of L2 norm ; one regularization option is to enforce
    # square of L2 norm to be small
    L2_sqr_out = (hidden_layer_w ** 2).sum() + (log_layer_w ** 2).sum()

    # Cost of hidden is the negative likelihood of hidden + the regularization
    # terms
    cost_hidden = neg_like_hidden \
        + L1_reg_term * L1_out \
        + L2_reg_term * L2_sqr_out

    # Cost of logistic is the negative likelihood of logistic + the regularization
    # terms
    cost_log = neg_like_log \
        + L1_reg_term * L1_out \
        + L2_reg_term * L2_sqr_out

    # Build out a list of gradients to calculate (weights and biases for hidden
    # & log)
    gparams = []
    gparams.append(T.grad(cost_hidden, hidden_layer_w))
    gparams.append(T.grad(cost_hidden, hidden_layer_b))
    gparams.append(T.grad(cost_log, log_layer_w))
    gparams.append(T.grad(cost_log, log_layer_b))

    return theano.function(
        [L1_reg_term, L2_reg_term, hidden_x, log_x, y,
            log_layer_w, log_layer_b, hidden_layer_w, hidden_layer_b], gparams)
