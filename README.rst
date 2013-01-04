Translated Version
==================

This repository is an attempt to translate the various demos contained in the
theano deep learning tutorial to regular python. It was done for two reasons:

  1. To help me learn theano
  2. To validate my understanding of the content of the tutorials

I've also found that it is useful to benchmark theano against my implementations.

Benchmark
=========

All benchmarks have been run on CPU not GPU.

Logistic SGD
------------

*Translated*
        Optimization complete with best validation score of 7.479167 %,with test performance 7.489583 %
        The code run for 75 epochs, with 1.674691 epochs/sec
        The code for file logistic_sgd.py ran for 44.8s

        real    0m29.138s
        user    0m45.541s
        sys     0m2.177s

*Theano*
        Optimization complete with best validation score of 7.479167 %,with test performance 7.489583 %
        The code run for 75 epochs, with 6.469746 epochs/sec
        The code for file logistic_sgd.py ran for 11.6s

        real    0m9.328s
        user    0m14.780s
        sys     0m1.116s

Deep Learning Tutorials
=======================

Deep Learning is a new area of Machine Learning research, which has been
introduced with the objective of moving Machine Learning closer to one of its
original goals: Artificial Intelligence.  Deep Learning is about learning
multiple levels of representation and abstraction that help to make sense of
data such as images, sound, and text.  The tutorials presented here will
introduce you to some of the most important deep learning algorithms and will
also show you how to run them using Theano.  Theano is a python library that
makes writing deep learning models easy, and gives the option of training them
on a GPU.

The easiest way to follow the tutorials is to `browse them online
<http://deeplearning.net/tutorial/>`_.

`Main development <http://github.com/lisa-lab/DeepLearningTutorials>`_
of this project.


Project Layout
--------------

Subdirectories:

- code - Python files corresponding to each tutorial
- data - data and scripts to download data that is used by the tutorials
- doc  - restructured text used by Sphinx to build the tutorial website
- html - built automatically by doc/Makefile, contains tutorial website
- issues_closed - issue tracking
- issues_open - issue tracking
- misc - administrative scripts


Build instructions
------------------

To build the html version of the tutorials, install sphinx and run doc/Makefile
