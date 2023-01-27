import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report,roc_auc_score,recall_score, accuracy_score
from logger import get_logger
from sklearn.model_selection import RepeatedStratifiedKFold
from read_data import read_data
from new_tree_Agglomerative import Decision_trees
import interpreter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif


def feature_select(x_train, y_train):

    sk = SelectKBest(chi2, k='all').fit(x_train, y_train)
    fea_index = np.argsort(- sk.scores_)

    return fea_index


for name in ["A549"]:
    data = read_data(name)
    data = pd.DataFrame(data)
    x = data.drop(data.shape[1]-1, axis=1)
    y = data[data.shape[1]-1]
    if type(y[0]) == str:
        classes = list(set(y))
        y = y.replace({classes[i]: i for i in range(len(classes))})
    x, y = np.array(x), np.array(y)
    print("*****************************{}_GINI***************************".format(name))
    print("Class distrbution: {}".format(dict(Counter(y))))
    print("data shape: {}".format(x.shape))
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
    k = 1
    count = {}
    for train_id, test_id in skf.split(x, y):
        print("----------Kfold-{}----------".format(k))
        k = k + 1
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

        count = interpreter.whole_tree_feature_importance(clf.tree, index, count)

    v = sorted(count.items(), key=lambda x: x[1], reverse=True)
    print(v)

