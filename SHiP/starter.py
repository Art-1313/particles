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

def get_metrics(X_train, X_test, y_train, y_test, cat_ft=[], vis=True, auc=False, proba=False):

    log_reg_auc = []
    rnd_frst_auc = []
    cat_bst_auc = []
    xg_bst_auc = []

    log_reg_proba = []
    rnd_frst_proba = []
    cat_bst_proba = []
    xg_bst_proba = []

    if vis:
        plt.figure(figsize=(25, 25))
        plt.suptitle('ROC curves for different classifiers', y=0.92)

    for i in range(len(states)):

        log_reg = LogisticRegression(n_jobs=-1, random_state=42, max_iter=10 ** 6).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        log_reg_auc.append(roc_auc_score(y_test[i], log_reg[:, 1]))
        log_reg_proba.append(log_reg[:, 1])

        if vis:
            plt.subplot(2, 2, 1)
            fpr, tpr, _ = roc_curve(y_test[i], log_reg[:, 1])
            plt.plot(fpr, tpr)
        
        cat_bst = CatBoostClassifier(verbose=False, random_state=42, cat_features=cat_ft).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        cat_bst_auc.append(roc_auc_score(y_test[i], cat_bst[:, 1]))
        cat_bst_proba.append(cat_bst[:, 1])

        if vis:
            plt.subplot(2, 2, 3)
            fpr, tpr, _ = roc_curve(y_test[i], cat_bst[:, 1])
            plt.plot(fpr, tpr)

        xg_bst = XGBClassifier().fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        xg_bst_auc.append(roc_auc_score(y_test[i], xg_bst[:, 1]))
        xg_bst_proba.append(xg_bst[:, 1])

        if vis:
            plt.subplot(2, 2, 4)
            fpr, tpr, _ = roc_curve(y_test[i], xg_bst[:, 1])
            plt.plot(fpr, tpr)

        rnd_frst = RandomForestClassifier(n_jobs=-1, random_state=42).fit(X_train[i], y_train[i]).predict_proba(X_test[i])
        rnd_frst_auc.append(roc_auc_score(y_test[i], rnd_frst[:, 1]))
        rnd_frst_proba.append(rnd_frst[:, 1])

        if vis:
            plt.subplot(2, 2, 2)
            fpr, tpr, _ = roc_curve(y_test[i], rnd_frst[:, 1])
            plt.plot(fpr, tpr, label='state {}'.format(states[i]))

    if vis:
        box = {'facecolor':'black', 'edgecolor': 'red', 'boxstyle': 'round'}

        plt.subplot(2, 2, 1)
        plt.title('Logistic Regression')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.text(0.8, 0.05, 'AUC = {:4f} ± {:4f}'.format(np.mean(log_reg_auc), np.std(log_reg_auc)), horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 3)
        plt.title('CatBoost')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.text(0.8, 0.05, 'AUC = {:4f} ± {:4f}'.format(np.mean(cat_bst_auc), np.std(log_reg_auc)), horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 4)
        plt.title('XGBoost')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.text(0.8, 0.05, 'AUC = {:4f} ± {:4f}'.format(np.mean(xg_bst_auc), np.std(rnd_frst_auc)), horizontalalignment = 'center', bbox = box, color = 'white')

        plt.subplot(2, 2, 2)
        plt.title('Randomn Forest')
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.text(0.8, 0.05, 'AUC = {:4f} ± {:4f}'.format(np.mean(rnd_frst_auc), np.std(rnd_frst_auc)), horizontalalignment = 'center', bbox = box, color = 'white')

        plt.legend(loc = 'upper right')

    if auc:
        return [np.mean(log_reg_auc), np.mean(rnd_frst_auc), np.mean(cat_bst_auc), np.mean(xg_bst_auc)]

    if proba:
        return [log_reg_proba, rnd_frst_proba, cat_bst_proba, xg_bst_proba]
    