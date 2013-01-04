import numpy


def neq(a, b):
    return numpy.invert(numpy.equal(a, b)).astype('int32')
