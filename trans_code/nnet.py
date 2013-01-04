import numpy


def softmax(w):
    w = numpy.array(w)

    maxes = numpy.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = numpy.exp(w - maxes)

    dist = e / numpy.sum(e, axis=1).reshape(maxes.shape[0], 1)
    return dist
