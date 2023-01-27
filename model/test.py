import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import classification_report,roc_auc_score,recall_score
from logger import get_logger
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score
from read_data import read_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from new_tree_Agglomerative import Decision_trees


def feature_select(x_train, y_train):

    sk = SelectKBest(chi2, k='all').fit(x_train, y_train)
    fea_index = np.argsort(- sk.scores_)

    return fea_index


LOGGER = get_logger("-", "new_tree__Agglomerative_SMOTE")
for name in ["HT29", "A375", "A549"]:
    data = read_data(name)
    data = pd.DataFrame(data)
    x = data.drop(data.shape[1]-1, axis=1)
    y = data[data.shape[1]-1]
    if type(y[0]) == str:
        classes = list(set(y))
        y = y.replace({classes[i]: i for i in range(len(classes))})
    x, y = np.array(x), np.array(y)
    LOGGER.info("*****************************{}_{}_50***************************".format(name, criterion))
    LOGGER.info("Class distrbution: {}".format(dict(Counter(y))))
    LOGGER.info("data shape: {}".format(x.shape))
    aucs, f1s, recall_0s, recall_1s, recall_2s, precision_0s, precision_1s, precision_2s = [], [], [], [], [], [], [], []
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
        index = feature_select(x_train, y_train)
        clf = Decision_trees(method="SMOTE")
        clf.fit(x_train[:, index[:50]], y_train)
        y_pred = clf.predict(x_test[:, index[:50]])
        y_pred_proba = clf.predict_proba(x_test[:, index[:50]])
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_pred_proba, average="macro")
        f1 = report['macro avg']['f1-score']
        recall_0 = report[str(list(dict(Counter(y)).keys())[0])]['recall']
        recall_1 = report[str(list(dict(Counter(y)).keys())[1])]['recall']
        recall_2 = report[str(list(dict(Counter(y)).keys())[2])]['recall']
        precision_0 = report[str(list(dict(Counter(y)).keys())[0])]['precision']
        precision_1 = report[str(list(dict(Counter(y)).keys())[1])]['precision']
        precision_2 = report[str(list(dict(Counter(y)).keys())[2])]['precision']
        aucs.append(auc)
        f1s.append(f1)
        recall_0s.append(recall_0)
        recall_1s.append(recall_1)
        recall_2s.append(recall_2)
        precision_0s.append(precision_0)
        precision_1s.append(precision_1)
        precision_2s.append(precision_2)
        k += 1

    LOGGER.info("AUC mean={:.4f}, std={:.4f}".format(np.mean(aucs), np.std(aucs)))
    LOGGER.info("f1_score mean={:.4f}, std={:.4f}".format(np.mean(f1s), np.std(f1s)))
    LOGGER.info("recall_0 mean={:.4f}, std={:.4f}".format(np.mean(recall_0s), np.std(recall_0s)))
    LOGGER.info("recall_1 mean={:.4f}, std={:.4f}".format(np.mean(recall_1s), np.std(recall_1s)))
    LOGGER.info("recall_2 mean={:.4f}, std={:.4f}".format(np.mean(recall_2s), np.std(recall_2s)))
    LOGGER.info("precision_0 mean={:.4f}, std={:.4f}".format(np.mean(precision_0s), np.std(precision_0s)))
    LOGGER.info("precision_1 mean={:.4f}, std={:.4f}".format(np.mean(precision_1s), np.std(precision_1s)))
    LOGGER.info("precision_2 mean={:.4f}, std={:.4f}".format(np.mean(precision_2s), np.std(precision_2s)))
