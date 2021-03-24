#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:11:31 2020

@author: Cecile Capponi
ONLY BINARY CLASSIFICATION IS CONCERNED, WITHIN REAL 2D INPUT SPACE
WEAK LEARNERS MUST BE STUMPS
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import timeit
import random
from matplotlib.path import Path
import matplotlib.patches as patches


# Surface of decision is a rectagle defined by learned stumps (at least 4)
# each rectangle will receive a color adapted to the class predicted for
# points in it.
class Rectangle:
    def __init__(self, xmin, xmax, ymin, ymax):  # limits of the rectangle
        self.ptopleft_ = (xmin, ymin)
        self.ptopright_ = (xmax, ymin)
        self.pbotleft_ = (xmin, ymax)
        self.pbotright_ = (xmax, ymax)
        self.center_ = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2])

    def __str__(self):
        return str(self.center_) + ' -- ' + str(self.class_)

    def set_class(self, c):
        self.class_ = c

    def tox(self):
        return [self.ptopleft_[0], self.pbotleft_[0],
                self.pbotright_[0], self.ptopright_[0]]

    def toy(self):
        return [self.ptopleft_[1], self.pbotleft_[1],
                self.pbotright_[1], self.ptopright_[1]]


# if considering the stump learned at iteration t of classifier clf, 
# this function returns on which component (0 or 1) the test was made
# and what is the threshold on that component
def getStump(clf, t):
    feature = clf.estimators_[t].tree_.feature[0]
    threshold = clf.estimators_[t].tree_.threshold[0]
    return [feature, threshold]


# Generates the rectangles of decisions
def generateZones(clf, limitsx, limitsy, T, process_all=1):
    # A list of two lists (thresholds on first then second component)
    TX = []
    TX.append([limitsx[0]])
    TX.append([limitsy[0]])
    # getting the weak separators given by stumps
    # On retourne la feature et le thresholds associé pour chaque itération et on le stocke
    # dans la bonne liste
    if process_all:
        for ite in range(T):
            stump = getStump(clf, ite)
            TX[stump[0]].append(stump[1])
    else:
        stump = getStump(clf, T)
        TX[stump[0]].append(stump[1])
    # on ajoute les limites max de x et y dans les listes correspondantes
    TX[0].append(limitsx[1])
    TX[1].append(limitsy[1])
    # sorting
    for i in [0, 1]:
        TX[i] = np.array(TX[i])
        TX[i].sort()
    # list of rectangles to be colored
    R = []
    # Pour chaque coordonées obtenues on crée un nouveau rectangle
    for yt in range(TX[1].shape[0] - 1):
        for xt in range(TX[0].shape[0] - 1):
            r = Rectangle(TX[0][xt], TX[0][xt + 1], TX[1][yt], TX[1][yt + 1])
            R.append(r)
    return R


def FindPoint(x1, y1, x2, y2, x, y):
    if (x >= x1 and x <= x2 and y >= y1 and y <= y2):
        return True
    else:
        return False


def setClassR(X0, X1, R):
    for r in R:
        sum0 = 0
        sum1 = 0
        for x in X0:
            if FindPoint(r.pbotleft_[0], r.pbotleft_[1], r.pbotright_[0], r.ptopright_[1], x[0], x[1]):
                sum0 += 1
        for x in X1:
            if FindPoint(r.pbotleft_[0], r.pbotleft_[1], r.pbotright_[0], r.ptopright_[1], x[0], x[1]):
                sum1 += 1
        if sum0 > sum1:
            r.set_class(0)
        elif sum0 < sum1:
            r.set_class(1)
        else:
            r.set_class(0)


def graphe(X, y, titre):
    bleu_x = []
    bleu_y = []
    rouge_x = []
    rouge_y = []

    for i in range(len(y)):
        if y[i]:
            bleu_x.append(X[i][0])
            bleu_y.append(X[i][1])
        else:
            rouge_x.append(X[i][0])
            rouge_y.append(X[i][1])

    plt.figure()
    plt.title("%s" % titre)
    plt.scatter(bleu_x, bleu_y, c='b')
    plt.scatter(rouge_x, rouge_y, c='r')
    plt.show()


def données_bruitées(X, y, beta):
    n = len(y)
    bis = int(beta * n / 100)
    label_mod = random.sample(np.arange(0, n, 1).tolist(), k=bis)
    for i in label_mod:
        if y[i]:
            y[i] = 0
        else:
            y[i] = 1
    return X, y

"""
X, Y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, class_sep=0.5, random_state=72)

clf = AdaBoostClassifier()
clf.fit(X, Y)
print(cross_val_score(clf, X, Y, cv=10).mean())

iter = [20, 50, 70, 90, 150, 300]

for i in iter:
    start = timeit.default_timer()
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(X, Y)
    stop = timeit.default_timer()
    print(str(i) + " iterations")
    print(str(stop - start) + ' s')
    scores = cross_val_score(clf, X, Y, cv=10)
    print("moyenne : " + str(scores.mean()))
    print("ecart-type : " + str(scores.std()) + '\n')

limitsx = [min(X[:, 0]), max(X[:, 0])]
limitsy = [min(X[:, 1]), max(X[:, 1])]

iter1 = [100, 400]
for i in iter1:
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(X, Y)
    y = clf.predict(X)
    X0 = []
    X1 = []
    for j in range(len(X)):
        if y[j] == 0:
            X0.append(X[j])
        else:
            X1.append(X[j])
    print(X0)
    print(X1)
    R = generateZones(clf, limitsx, limitsy, i)
    setClassR(X0, X1, R)
    Y = ['r' if y != 0 else 'b' for y in Y]
    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    for r in R:
        width = r.pbotright_[0] - r.pbotleft_[0]
        height = r.ptopleft_[1] - r.pbotleft_[1]
        if r.class_ == 0:
            patch = patches.Rectangle(r.pbotleft_, width, height, alpha=0.5, color='b')
        else:
            patch = patches.Rectangle(r.pbotleft_, width, height, alpha=0.5, color='r')
        ax.add_patch(patch)
    plt.show()

Y = ['r' if y != 0 else 'b' for y in Y]

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
"""
print("cas extreme\n")


XHA = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4],
                [1, 3], [1, 4], [2, 3], [2, 4], [3, 1], [3, 2], [4, 1], [4, 2]])
YHA = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

XHB = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4],
                [1, 3], [1, 4], [2, 3], [2, 4], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1]])
YHB = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

XHC = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4],
                [1, 3], [1, 4], [2, 3], [2, 4], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1]])
YHC = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1])

print("\nQuestion 1:\n")

print("DATASET A")
for i in range(1, 20):
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(XHA, YHA)
    print("Pour", i, "itérations.\nTemps d'apprentissage:")
    print("Le score est:", cross_val_score(clf, XHA, YHA, cv=5))

graphe(XHA, YHA, "Dataset A")

print("\nDATASET B")
for i in range(10, 20):
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(XHB, YHB)
    print("Pour", i, "itérations.\nTemps d'apprentissage:")
    print("Le score est:", cross_val_score(clf, XHB, YHB, cv=2).mean())

graphe(XHB, YHB, "Dataset B")

print("\nDATASET C")
for i in range(10, 20):
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(XHC, YHC)
    print("Pour", i, "itérations.\nTemps d'apprentissage:")
    print("Le score est:", cross_val_score(clf, XHC, YHC, cv=5).mean())

graphe(XHC, YHC, "Dataset C")

score_cv = []
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, class_sep=0.5,
                           random_state=72)

iteration = np.arange(10, 55, 5)
for beta in iteration:
    clf = AdaBoostClassifier(n_estimators=50)
    Xd, yd = données_bruitées(X, y, beta)
    # graphe(X,y, 'Bruit %i' % beta)
    score_cv.append(1 - np.mean(cross_val_score(clf, Xd, yd, cv=5)))

plt.figure()
axes = plt.gca()
axes.set_xlabel('Bruit en pourcentage')
axes.set_ylabel("Estimation de l'erreur")
plt.title("Evolution de la fonction d'erreur en fonction du bruit")
plt.plot(iteration, score_cv, c='b')
plt.show()
