import matplotlib.pylab as plt
import sys
import numpy as np
import pandas as pd

import itertools as it

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

sns.set(font_scale=1.5, style='whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from itertools import combinations

from xgboost import XGBClassifier

############

states = [0, 6, 7, 13, 42, 911]

############

data_bkg = np.load('shield_cands_muons.npz')
w_bkg = data_bkg['W']
X_bkg = pd.DataFrame({'X': data_bkg['X'], 'Y': data_bkg['Y'], 'Px': data_bkg['Px'], 'Py': data_bkg['Py'], 'Pz': data_bkg['Pz']})

data_sig = np.load('HNL_100k_ecn3_geom_1_4_ch_muons.npz')
w_sig = data_sig['W']
X_sig = pd.DataFrame({'X': data_sig['X'], 'Y': data_sig['Y'], 'Px': data_sig['Px'], 'Py': data_sig['Py'], 'Pz': data_sig['Pz']})

data = X_bkg.append(X_sig)
data['type'] = np.hstack((np.ones_like(w_bkg), np.zeros_like(w_sig)))

############

def print_err(a, a_err, vis=True):
    a = np.array([a]).flatten()
    a_err = np.array([a_err]).flatten()
    val = np.zeros_like(a)
    sig = np.zeros_like(a_err)
    for i in range(a.size):
        val[i] = np.round(a[i], decimals = round(1 * (np.trunc(np.log10(a_err[i]) < 0)) - np.trunc(np.log10(a_err[i]))))
        sig[i] = np.round(a_err[i], decimals = round(1 * (np.trunc(np.log10(a_err[i]) < 0)) - np.trunc(np.log10(a_err[i]))))
        if vis: print(val[i], '\pm', sig[i])
    return val, sig

############

def get_metrics(X_train, X_test, y_train, y_test, cat_ft=[], vis=True, auc=False, proba=False):

    log_reg_auc = []
    rnd_frst_auc = []
    cat_bst_auc = []
    xg_bst_auc = []

    log_reg_proba = []
    rnd_frst_proba = []
    cat_bst_proba = []
    xg_bst_proba = []

    scale = StandardScaler()

    cols = list(X_train[0].columns.values)

    for x in cat_ft: cols.remove(x)

    if vis:
        plt.figure(figsize=(25, 25))

    for i in range(len(states)):

        scale.fit(X_train[i][cols])

        X_train[i][cols] = scale.transform(X_train[i][cols])
        X_test[i][cols] = scale.transform(X_test[i][cols])

        log_reg = LogisticRegression(n_jobs=-1, random_state=42, max_iter=10 ** 6).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        log_reg_auc.append(roc_auc_score(y_test[i], log_reg[:, 1]))
        log_reg_proba.append(log_reg[:, 1])

        if vis:
            plt.subplot(2, 2, 1)
            fpr, tpr, _ = roc_curve(y_test[i], log_reg[:, 1])
            plt.plot(fpr, tpr)
        
        cat_bst = CatBoostClassifier(verbose=False, random_state=42, cat_features=cat_ft, n_estimators=2000).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        cat_bst_auc.append(roc_auc_score(y_test[i], cat_bst[:, 1]))
        cat_bst_proba.append(cat_bst[:, 1])

        if vis:
            plt.subplot(2, 2, 3)
            fpr, tpr, _ = roc_curve(y_test[i], cat_bst[:, 1])
            plt.plot(fpr, tpr)

        xg_bst = XGBClassifier(n_jobs=-1).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        xg_bst_auc.append(roc_auc_score(y_test[i], xg_bst[:, 1]))
        xg_bst_proba.append(xg_bst[:, 1])

        if vis:
            plt.subplot(2, 2, 4)
            fpr, tpr, _ = roc_curve(y_test[i], xg_bst[:, 1])
            plt.plot(fpr, tpr)

        rnd_frst = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=2000, min_samples_leaf=5).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        rnd_frst_auc.append(roc_auc_score(y_test[i], rnd_frst[:, 1]))
        rnd_frst_proba.append(rnd_frst[:, 1])

        if vis:
            plt.subplot(2, 2, 2)
            fpr, tpr, _ = roc_curve(y_test[i], rnd_frst[:, 1], drop_intermediate=False)
            plt.plot(fpr, tpr, label='state {}'.format(states[i]))

    if vis:
        box = {'facecolor':'black', 'edgecolor': 'red', 'boxstyle': 'round'}

        plt.subplot(2, 2, 1)
        plt.title('Logistic Regression', fontdict={'size': 50, 'fontname': 'Times New Roman'})
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('Fake rate')
        plt.ylabel('Efficiency')
        val, err = print_err(np.mean(log_reg_auc), np.std(log_reg_auc), vis=False)
        plt.text(0.7, 0.05, 'AUC = {} ± {}'.format(val[0], err[0]), fontdict={'size': 40, 'fontname': 'Times New Roman'}, horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 3)
        plt.title('CatBoost', fontdict={'size': 50, 'fontname': 'Times New Roman'})
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('Fake rate')
        plt.ylabel('Efficiency')
        val, err = print_err(np.mean(cat_bst_auc), np.std(cat_bst_auc), vis=False)
        plt.text(0.7, 0.05, 'AUC = {} ± {}'.format(val[0], err[0]), fontdict={'size': 40, 'fontname': 'Times New Roman'}, horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 4)
        plt.title('XGBoost', fontdict={'size': 50, 'fontname': 'Times New Roman'})
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('Fake rate')
        plt.ylabel('Efficiency')
        val, err = print_err(np.mean(xg_bst_auc), np.std(xg_bst_auc), vis=False)
        plt.text(0.7, 0.05, 'AUC = {} ± {}'.format(val[0], err[0]), fontdict={'size': 40, 'fontname': 'Times New Roman'}, horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 2)
        plt.title('Random Forest', fontdict={'size': 50, 'fontname': 'Times New Roman'})
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('Fake rate')
        plt.ylabel('Efficiency')
        val, err = print_err(np.mean(rnd_frst_auc), np.std(rnd_frst_auc), vis=False)
        plt.text(0.7, 0.05, 'AUC = {} ± {}'.format(val[0], err[0]), fontdict={'size': 40, 'fontname': 'Times New Roman'}, horizontalalignment = 'center', bbox = box, color = 'white')

        plt.legend(loc = 'upper right')

    if auc:
        return [np.mean(log_reg_auc), np.mean(rnd_frst_auc), np.mean(cat_bst_auc), np.mean(xg_bst_auc)]

    if proba:
        return [log_reg_proba, rnd_frst_proba, cat_bst_proba, xg_bst_proba]
    