#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = '3404199'

import numpy as np
import Parser
import itertools
import Tools
from Models import *
from multiprocessing import Pool


def evalSVDGD(config):
    pairs_lt = Tools.folder2pairs(r"../ml-100k/")
    m_score = 0.0
    for pair in pairs_lt:
        fgd = SVDGD(pair[0], epochs=config[0], nZ=config[1], lambda_4=config[2], learning_rate=config[3])
        try:
            score = fgd.predict(Parser.MatrixParser.txttocsc(pair[1]))
            m_score += score
        except FloatingPointError:
            score = -100
            m_score += score
            print "FP ERROR", pair[2], config
            break
        except:
            score = -500
            m_score += score
            print "PREDICT ERROR", pair[2], config
            break
        # m_score += score

    m_score /= len(pairs_lt)
    print "{:2d}, {:3d}, {:2.4f}, {:2.4f}, {:2.4f}".format(config[0], config[1], config[2], config[3], m_score)
    return m_score


def main():
    # Configs de test
    epochs = [10]
    nZ = [2, 5, 10, 20, 50, 100, 200]
    lambda_4 = 10 ** np.arange(-4, 0, 1, dtype=np.float64)
    learning_rate = 10 ** np.arange(-4, 0, 1, dtype=np.float64)

    configs = list(itertools.product(epochs, nZ, lambda_4, learning_rate))
    print "n configs: ", len(configs)
    # Loop en parallele
    nthreads = 8
    parallel_pool = Pool(nthreads)
    scores = []
    try:
        scores = parallel_pool.map(evalSVDGD, configs)
    except KeyboardInterrupt:
        print "Interrupt from keyboard, saving log"

    with open("log_svdgd.csv", 'w') as f:
        f.write("epochs,nZ,lambda_4,learning_rate,score\n")
        for i in xrange(len(configs)):
            config = configs[i]
            f.write("{},{},{},{},{}\n".format(config[0], config[1], config[2], config[3], scores[i]))


if __name__ == '__main__':
    main()