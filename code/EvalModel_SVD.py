#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = '3404199'

import numpy as np
import Parser
import itertools
import Tools
from Models import *
from multiprocessing import Pool


def evalSVD(config):
    pairs_lt = Tools.folder2pairs(r"../ml-100k/")
    m_score = 0.0
    for pair in pairs_lt:
        svd = SVD(pair[0], k=config[0])
        try:
            score = svd.predict(Parser.MatrixParser.txttocsc(pair[1]))
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
    print "{:2d}, {:2.4f}".format(config[0], m_score)
    return m_score


def main():
    # Configs de test
    nZ = [2, 5, 10, 20, 50, 100, 200, 400, 800]

    configs = list(itertools.product(nZ))
    print "n configs: ", len(configs)
    # Loop en parallele
    nthreads = 7
    parallel_pool = Pool(nthreads)
    scores = []
    try:
        scores = parallel_pool.map(evalSVD, configs)
    except KeyboardInterrupt:
        print "Interrupt from keyboard, saving log"

    with open("log_svd.csv", 'w') as f:
        f.write("nZ,score\n")
        for i in xrange(len(configs)):
            config = configs[i]
            f.write("{},{}\n".format(config[0], scores[i]))


if __name__ == '__main__':
    main()