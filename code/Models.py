#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle

__author__ = '3404199'

import numpy as np
import scipy
import scipy.stats
import scipy.sparse.linalg
import Parser
import itertools
import Tools
import math
import matplotlib.pyplot as plt


# ub  ~ 1.08
# Ib  ~ 1.05
# sim ~ 0.6

class Bias:
    def __init__(self, data, out_name='item.png', case='item'):
        self.data = data.copy()
        self.case = case
        if case == 'item':
            self.pred = self.data.sum(2).flatten()
        else:
            self.pred = self.data.sum(1).flatten()

        self.m_mean = self.data.sum()/self.data.nnz

        old_conf = np.seterr(divide='ignore')
        if case == 'item':
            cases = self.data.getnnz(0).flatten()
            cases[cases == 0] = 1
            self.pred = np.divide(self.pred, cases)
        else:
            cases = self.data.getnnz(1).flatten()
            cases[cases == 0] = 1
            self.pred = np.divide(self.pred, cases)
        np.seterr(**old_conf)

        self.pred[np.isnan(self.pred)] = self.m_mean

        plt.figure()
        plt.hist(self.pred.T, 100)
        if case == 'item':
            plt.title('Item')
        else:
            plt.title('User')

        plt.savefig(out_name)

    def predict(self, test):
        cx = test.tocoo()
        score = 0.0
        if self.case == 'item':
            for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
                score += (self.score_item(item) - n) ** 2
        else:
            for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
                score += (self.score_item(user) - n) ** 2
        return math.sqrt(score/cx.nnz)

    def score_item(self, item):
        try:
            return self.pred[0, item]
        except IndexError:
            return self.m_mean


class PBias:
    def __init__(self, filename):
        data = np.loadtxt(filename)

        maxs = data.max(0) # extraction des nb d'utilisateurs et d'item
        nu = maxs[0]+1
        ni = maxs[1]+1

        self.user_bias = np.zeros(nu)
        self.item_bias = np.zeros(ni)
        user_count = np.zeros(nu)
        item_count = np.zeros(ni)
        for iteration in xrange(len(data)):
            u = data[iteration,0]
            i = data[iteration,1]
            r = data[iteration,2]
            self.user_bias[u] += r
            self.item_bias[i] += r
            user_count[u] += 1
            item_count[i] += 1
        self.user_bias /= np.where(user_count == 0, 1, user_count)
        self.item_bias /= np.where(item_count == 0, 1, item_count)

    def predict(self, test):
        score = 0.0
        data = np.loadtxt(test)
        for iteration in xrange(len(data)):
            u = data[iteration, 0]
            i = data[iteration, 1]
            r = data[iteration, 2]
            score += (self.score_item(i) - r) ** 2

        return math.sqrt(score/len(data))

    def score_item(self, item):
        return self.item_bias[item]


class SimiTriple:
    def __init__(self, filen):
        data = np.loadtxt(filen)    # chargement du fichier
        pbias = PBias(filen)
        self.mdata = Parser.MatrixParser.txttocsc(filen)
        self.user_bias = pbias.user_bias

        maxs = data.max(0)
        nu = int(maxs[0]+1)
        ni = int(maxs[1]+1)

        self.sim = np.zeros((nu, nu))
        den1 = np.zeros((nu, nu))
        den2 = np.zeros((nu, nu))

        for i in xrange(1, ni):
            if i % 400 == 0:
                print i
            subdata = data[data[:, 1] == i, :]  # selection du produit i
            n = subdata.shape[0]
            for i1 in xrange(n):
                for i2 in xrange(i1, n):
                    u1 = subdata[i1, 0]
                    u2 = subdata[i2, 0]
                    r1 = subdata[i1, 2]
                    r2 = subdata[i2, 2]
                    v1 = (r1-self.user_bias[u1])
                    v2 = (r2-self.user_bias[u2])
                    self.sim[u1, u2] += v1*v2
                    self.sim[u2, u1] += v1*v2
                    den1[u1, u2] += v1**2
                    den1[u2, u1] += v1**2
                    den2[u1, u2] += v2**2
                    den2[u2, u1] += v2**2

        self.sim /= np.maximum(np.sqrt(den1)*np.sqrt(den2), 1)

    def score_item(self, user, item):
        pred = self.user_bias[user]
        #den = np.sum(self.sim[user, :]) - self.sim[user, user]
        den = 0.0
        cx = self.mdata.getcol(item).tocoo()
        num = 0.0
        for u, i, v in itertools.izip(cx.row, cx.col, cx.data):
            if u == user:
                continue
            num += self.sim[user, u]*(v - self.user_bias[u])
            den += self.sim[user, u]
        try:
            pred += num/den
        except ZeroDivisionError:
            pass
        return pred

    def predict(self, test):
        cx = test.tocoo()
        score = 0.0

        for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
            score += (self.score_item(user, item) - n) ** 2
        return math.sqrt(score/cx.nnz)


class SVD:
    def __init__(self, filen, k=10):

        A = Parser.MatrixParser.txttocsc(filen)

        self.user_bias = A.sum(1).flatten()
        rated_items = A.getnnz(1).flatten()
        rated_items[rated_items == 0] = 1
        self.user_bias = np.divide(self.user_bias, rated_items)

        self.U, St, self.I = scipy.sparse.linalg.svds(A, k=k)
        self.S = scipy.sparse.diags(St, 0)

    def score_item(self, user, item):
        try:
            score = self.U[user, :].dot(self.S.dot(self.I[:, item]))
        except IndexError:
            score = 0.0
        score += self.user_bias[0, user]
        return min(5.0, max(0, score))

    def predict(self, test):
        cx = test.tocoo()
        score = 0.0

        for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
            score += (self.score_item(user, item) - n) ** 2
        return math.sqrt(score/cx.nnz)


# factorisation matricielle: on travaille de nouveau sur les triplets pour minimiser les hypothèses
# sur les cases vides

class FGD:
    def __init__(self, filen, epochs=5, nZ=10, l1_weight=0.00, l2_weight=0.0001, learning_rate=0.001):
        
        data = np.loadtxt(filen)
        maxs = data.max(0) # extraction des nb d'utilisateurs et d'item
        nu = maxs[0]+1
        ni = maxs[1]+1
        self.user_bias = np.zeros(nu)
        user_count = np.zeros(nu)
        
        for iteration in xrange(len(data)):
            u = data[iteration, 0]
            r = data[iteration, 2]
            self.user_bias[u] += r
            user_count[u] += 1
        self.user_bias /= np.where(user_count == 0, 1, user_count)

        random = np.random.RandomState(0)
        #epochs = 5 # nb de passage sur la base
        # nZ = 10 # taille de l'espace latent
        # l1_weight = 0.00 # contraintes de régularization L1 + L2
        # l2_weight = 0.0001
        # learning_rate = 0.001

        train_indexes = np.arange(len(data)) # pour cet exemple, je prends tous les indices en apprentissage...

        # initialisation à moitié vide... randn + seuillage > 0
        self.user_latent = np.random.randn(nu, nZ)
        self.item_latent = np.random.randn(ni, nZ)
        self.user_latent = np.where(self.user_latent > 0, self.user_latent, 0) # profils positifs sparses
        self.item_latent = np.where(self.item_latent > 0, self.item_latent, 0)

        for epoch in xrange(epochs):
            #print "epoch : %d"%epoch
            # Update
            random.shuffle(train_indexes)
            for index in train_indexes:
                # extraction des variables => lisibilité
                label, user, item = data[index,2], data[index,0], data[index,1]
                gamma_u, gamma_i = self.user_latent[user, :], self.item_latent[item, :]
                # Optimisation
                delta_label = 2 * (label - np.dot(gamma_u, gamma_i))
                gradient_u = l2_weight * gamma_u + l1_weight - delta_label * gamma_i
                gamma_u_prime = gamma_u - learning_rate * gradient_u
                self.user_latent[user, :] = np.where(gamma_u_prime * gamma_u > 0, gamma_u_prime, 0) # MAJ user
                gradient_i = l2_weight * gamma_i + l1_weight - delta_label * gamma_u
                gamma_i_prime = gamma_i - learning_rate * gradient_i
                self.item_latent[item, :] = np.where(gamma_i_prime * gamma_i > 0, gamma_i_prime, 0) # MAJ item

    def score_item(self, user, item):
        try:
            score = np.dot(self.user_latent[user, :], self.item_latent[item, :])
        except IndexError:
            score = 0.0
        score += self.user_bias[user]
        return min(5.0, max(0, score))

    def predict(self, test):
        cx = test.tocoo()
        score = 0.0

        for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
            score += (self.score_item(user, item) - n) ** 2
        return math.sqrt(score/cx.nnz)


class SVDGD:
    def __init__(self, filen, nZ=10, learning_rate=0.005, lambda_4=0.02, epochs=5):
        data = np.loadtxt(filen)
        maxs = data.max(0) # extraction des nb d'utilisateurs et d'item
        nu = maxs[0]+1
        ni = maxs[1]+1

        self.mi = 0

        for iteration in xrange(len(data)):
            self.mi += data[iteration, 2]
        self.mi /= len(data)

        random = np.random.RandomState(0)
        train_indexes = np.arange(len(data)) # pour cet exemple, je prends tous les indices en apprentissage...

        # initialisation à moitié vide... randn + seuillage > 0
        self.user_bias = np.random.randn(nu)
        self.item_bias = np.random.randn(ni)
        self.user_latent = np.random.randn(nu, nZ)
        self.item_latent = np.random.randn(ni, nZ)
        self.user_latent = np.where(self.user_latent > 0, self.user_latent, 0) # profils positifs sparses
        self.item_latent = np.where(self.item_latent > 0, self.item_latent, 0)
        # old = np.seterr(all='raise')
        for epoch in xrange(epochs):
            #print "epoch : %d"%epoch
            # Update
            random.shuffle(train_indexes)
            for index in train_indexes:
                # extraction des variables => lisibilité
                score, user, item = data[index, 2], data[index, 0], data[index, 1]
                pscore = self.mi + self.user_bias[user] + self.item_bias[item] + np.dot(self.user_latent[user, :],
                                                                                        self.item_latent[item, :])
                e = np.float64(score - pscore)
                self.user_bias[user] += learning_rate*(e - lambda_4*self.user_bias[user])
                self.item_bias[item] += learning_rate*(e - lambda_4*self.item_bias[item])
                old_il = np.copy(self.item_latent[item, :])
                old_ul = np.copy(self.user_latent[user, :])
                self.item_latent[item, :] += learning_rate*(e*old_ul - lambda_4*old_il)
                self.user_latent[user, :] += learning_rate*(e*old_il - lambda_4*old_ul)
        # np.seterr(**old)
    def score_item(self, user, item):
        score = self.mi
        try:
            score += self.user_bias[user]+self.item_bias[item] + np.dot(self.user_latent[user, :], self.item_latent[item, :])
        except IndexError:
            pass
        return score

    def predict(self, test):
        cx = test.tocoo()
        score = 0.0
        old = np.seterr(all='raise')
        try:
            for user, item, n in itertools.izip(cx.row, cx.col, cx.data):
                score += (self.score_item(user, item) - n) ** 2
        except FloatingPointError as err:
            np.seterr(**old)
            raise err
        np.seterr(**old)
        return math.sqrt(score/cx.nnz)


def main():
    # learn_data = Parser.MatrixParser.txttocsc(r"../ml-100k/u1.base")
    # test_data = Parser.MatrixParser.txttocsc(r"../ml-100k/u1.test")
    pairs_lt = Tools.folder2pairs(r"../ml-100k/")
    m_score = 0.0
    for pair in pairs_lt:
        # bias = Bias(Parser.MatrixParser.txttocsc(pair[0]), "item_{}.png".format(pair[2]), case='item')
        # score = bias.predict(Parser.MatrixParser.txttocsc(pair[1]))

        # export_name = "{}.sim".format(pair[0].split("/")[-1])
        #
        # if os.path.isfile(export_name):
        #     bfile = open(export_name, 'rb')
        #     sim = pickle.load(bfile)
        #     bfile.close()
        # else:
        #     sim = SimiTriple(pair[0])
        #     filehandler = open(export_name, "wb")
        #     pickle.dump(sim, filehandler)
        #     filehandler.close()

        # svd = SVD(pair[0], k=10)
        # score = svd.predict(Parser.MatrixParser.txttocsc(pair[1]))

        # fgd = FGD(pair[0], epochs=12, nZ=10, l1_weight=0.1, l2_weight=0.1, learning_rate=0.01)
        # score = fgd.predict(Parser.MatrixParser.txttocsc(pair[1]))

        svdgd = SVDGD(pair[0], nZ=2, learning_rate=0.1, lambda_4=10, epochs=10)
        score = svdgd.predict(Parser.MatrixParser.txttocsc(pair[1]))

        print pair[2], score
        m_score += score
        break
    print "Mean", m_score/len(pairs_lt)


if __name__ == "__main__":
    main()