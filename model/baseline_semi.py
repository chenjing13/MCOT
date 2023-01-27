import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import Counter
from sklearn.metrics import classification_report,roc_auc_score,recall_score
from logger import get_logger
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score
from read_data import read_data
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from itertools import combinations
from sklearn.feature_selection import SelectKBest, chi2
from new_tree_Agglomerative_Semi import Decision_trees
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from sklearn.preprocessing import OneHotEncoder


def feature_select(x_train, y_train):

    sk = SelectKBest(chi2, k='all').fit(x_train, y_train)
    fea_index = np.argsort(- sk.scores_)

    return fea_index


LOGGER = get_logger("-", "baseline_Semi_0.8")
for name in ["HT29", "A375", "A549"]:
    data = read_data(name)
    data = pd.DataFrame(data)
    x = data.drop(data.shape[1] - 1, axis=1)
    y = data[data.shape[1] - 1]
    if type(y[0]) == str:
        classes = list(set(y))
        y = y.replace({classes[i]: i for i in range(len(classes))})
    x, y = np.array(x), np.array(y)
    for tag in ["LP", "LS", "MYT", "Assemble", "CoTraining", "TriTraining"]:
        LOGGER.info("*****************************{}_{}***************************".format(name, tag))
        LOGGER.info("Class distrbution: {}".format(dict(Counter(y))))
        LOGGER.info("data shape: {}".format(x.shape))
        f1s, recall_0s, recall_1s, recall_2s, precision_0s, precision_1s, precision_2s = [], [], [], [], [], [], []
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
        k = 1
        for train_id, test_id in skf.split(x, y):
            LOGGER.info("----------Kfold-{}----------".format(k))
            x_train, x_test = x[train_id], x[test_id]
            y_train, y_test = y[train_id], y[test_id]
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_train = np.array(x_train, type(float))
            x_test = scaler.transform(x_test)
            x_test = np.array(x_test, type(float))
            know_x, unknow_x, know_y, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
            index = feature_select(know_x, know_y)
            unknow_y = np.array([-1 for i in range(len(unknow_x))])
            x_train = np.vstack((know_x, unknow_x))
            y_train = np.hstack((know_y, unknow_y))
            if tag == "LP":
                clf = LabelPropagation()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
            elif tag == "LS":
                clf = LabelSpreading()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
            elif tag == "MYT":
                clf = Decision_trees(method="SMOTE")
                clf.fit(x_train[:, index[:50]], y_train)
                y_pred = clf.predict(x_test[:, index[:50]])
            elif tag == "Assemble":
                clf = Assemble()
                clf.fit(X=know_x, y=know_y, unlabeled_X=unknow_x)
                y_pred = clf.predict(x_test)
            elif tag == "CoTraining":
                clf = Co_Training()
                clf.fit(X=know_x, y=know_y, unlabeled_X=unknow_x)
                y_pred = clf.predict(x_test)
            elif tag == "TriTraining":
                clf = Tri_Training()
                clf.fit(X=know_x, y=know_y, unlabeled_X=unknow_x)
                y_pred = clf.predict(x_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report['macro avg']['f1-score']
            recall_0 = report[str(list(dict(Counter(y)).keys())[0])]['recall']
            recall_1 = report[str(list(dict(Counter(y)).keys())[1])]['recall']
            recall_2 = report[str(list(dict(Counter(y)).keys())[2])]['recall']
            precision_0 = report[str(list(dict(Counter(y)).keys())[0])]['precision']
            precision_1 = report[str(list(dict(Counter(y)).keys())[1])]['precision']
            precision_2 = report[str(list(dict(Counter(y)).keys())[2])]['precision']
            k += 1
            f1s.append(f1)
            recall_0s.append(recall_0)
            recall_1s.append(recall_1)
            recall_2s.append(recall_2)
            precision_0s.append(precision_0)
            precision_1s.append(precision_1)
            precision_2s.append(precision_2)

        LOGGER.info("f1_score mean={:.4f}, std={:.4f}".format(np.mean(f1s), np.std(f1s)))
        LOGGER.info("recall_0 mean={:.4f}, std={:.4f}".format(np.mean(recall_0s), np.std(recall_0s)))
        LOGGER.info("recall_1 mean={:.4f}, std={:.4f}".format(np.mean(recall_1s), np.std(recall_1s)))
        LOGGER.info("recall_2 mean={:.4f}, std={:.4f}".format(np.mean(recall_2s), np.std(recall_2s)))
        LOGGER.info("precision_0 mean={:.4f}, std={:.4f}".format(np.mean(precision_0s), np.std(precision_0s)))
        LOGGER.info("precision_1 mean={:.4f}, std={:.4f}".format(np.mean(precision_1s), np.std(precision_1s)))
        LOGGER.info("precision_2 mean={:.4f}, std={:.4f}".format(np.mean(precision_2s), np.std(precision_2s)))
