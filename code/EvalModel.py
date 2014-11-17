#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = '3404199'

import numpy as np
import Parser
import itertools
import Tools
from Models import *
from multiprocessing import Pool

def evalFGD(config):
    pairs_lt = Tools.folder2pairs(r"../ml-100k/")
    m_score = 0.0
    for pair in pairs_lt:
        fgd = FGD(pair[0], epochs=config[0], nZ=config[1], l1_weight=config[2], l2_weight=config[3],
                  learning_rate=config[4])
        try:
            score = fgd.predict(Parser.MatrixParser.txttocsc(pair[1]))
        except:
            score = 0
            print "PREDICT ERROR", pair[2], config
        m_score += score

    m_score /= len(pairs_lt)
    print "{:2d}, {:3d}, {:2.4f}, {:2.4f}, {:2.4f}, {:2.4f}".format(config[0], config[1], config[2], config[3], config[4],
                                                              m_score)
    return m_score


def main():
    # Configs de test
    epochs = [10]
    nZ = [2, 5, 10, 25, 50, 100, 200, 400]
    l1_weight = 10 ** np.arange(-4, 2, 1, dtype=np.float64)
    l2_weight = 10 ** np.arange(-4, 2, 1, dtype=np.float64)
    learning_rate = 10 ** np.arange(-4, 1, 1, dtype=np.float64)

    configs = list(itertools.product(epochs, nZ, l1_weight, l2_weight, learning_rate))

    # Loop en parallele
    nthreads = 7
    parallel_pool = Pool(nthreads)
    scores = []
    try:
        scores = parallel_pool.map(evalFGD, configs)
    except KeyboardInterrupt:
        print "Interrupt from keyboard, saving log"

    with open("log.csv", 'w') as f:
        f.write("epochs,nZ,l1_weight,l2_weight,learning_rate,score\n")
        for i in xrange(len(configs)):
            config = configs[i]
            f.write("{},{},{},{},{},{}\n".format(config[0], config[1], config[2], config[3], config[4], scores[i]))


if __name__ == '__main__':
    main()