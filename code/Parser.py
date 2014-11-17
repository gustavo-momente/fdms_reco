 #!/usr/bin/python
 # -*- coding: utf-8 -*-

__author__ = 'Usuário'

import numpy as np
import scipy.sparse


class MatrixParser:
    @staticmethod
    def txttocsc(in_file, usecols=(0, 1, 2)):
        data = np.loadtxt(in_file, usecols=usecols)
        sparse_rep = scipy.sparse.csc_matrix(scipy.sparse.coo_matrix((data[:, 2], (data[:, 0], data[:, 1]))))
        return sparse_rep

    @staticmethod
    def txt2user_bias(in_file):
        data = MatrixParser.txttocsc(in_file)
        user_bias = data.mean(axis=1).flatten()
        return user_bias