from __future__ import division, absolute_import
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from itertools import chain
from collections import Counter
from sklearn import metrics


def up_sampling(data, percent=1):

    data.iloc[:, -1] = data.iloc[:, -1].astype("int")
    class_y = Counter((list(data.iloc[:, -1])))
    max_key = max(class_y, key=class_y.get)
    for ky in class_y.keys():
        if ky != max_key:
            data1 = data[data['y'] == max_key]
            data0 = data[data['y'] == ky]
            index = np.random.randint(
                len(data0), size=percent * (len(data1) - len(data0)))
            up_data0 = data0.iloc[list(index)]
            data = pd.concat([data, up_data0])
    return data


def gini_index(groups, groups_values):

    gini = 0.0
    number = 0.0
    for group in groups:
        number += len(group)
    for i, group in enumerate(groups):
        size = sum([1 for row in group])
        if size == 0:
            continue
        else:
            proportion = sum([1 for row in group if row[-1] == groups_values[i]]) / float(size)
            gini_group = 1 - (proportion ** 2) - ((1.0 - proportion) ** 2)
            gini += gini_group * len(group) / number

    return gini


def c45_index(groups, groups_values):

    entropy = 0.0
    number = 0.0
    all_group = []
    for group in groups:
        number += len(group)
        all_group += group
    for value in list(set(groups_values)):
        proportion = sum([row[-1] for row in all_group if row[-2] == value]) / float(sum(row[-1] for row in all_group))
        if proportion == 0.0:
            continue
        else:
            entropy -= proportion * np.log2(proportion)
    entropy_groups = 0.0
    splitInfo = 0.0
    for i, group in enumerate(groups):
        size = sum(row[-1] for row in group)
        if size == 0:
            continue
        proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
        if proportion == 1.0 or proportion == 0.0:
            entropy_group = 0.0
        else:
            entropy_group = -proportion * np.log2(proportion)-(1-proportion) * np.log2(1-proportion)
        splitInfo -= (len(group) / number) * np.log2(len(group) / number)
        entropy_groups += entropy_group * len(group) / number
    entropy = entropy - entropy_groups
    if splitInfo == 0:
        return entropy
    else:
        return entropy/splitInfo


class tree_stump(object):

    def __init__(self):
        self.n_samples = 0
        self.feature = None
        self.train_data = None
        self.tree = None
        self.best_n_clusters = None

    def fit(self, data, seed):

        self.n_samples = data.shape[0]
        if len(list(set(data.iloc[:, -1].astype("int")))) == 1:
            self.tree = {"score": list(set(data.iloc[:, -1].astype("int")))[0], "c_label": []}
        else:
            self.tree, self.feature, self.train_data, self.best_n_clusters = self.build_tree_stump(data, [seed])

        return self

    def get_split(self, c, data):

        groups = [[]for i in range(len(set(c)))]
        core = list(set(c))
        for i, row in enumerate(data):
            groups[core.index(c[i])].append(row)
        proportion = []
        groups_value = []
        for group in groups:
            proportion.append(dict(Counter([row[-1] for row in group])))
            groups_value.append(max(dict(Counter([row[-1] for row in group])), key=dict(Counter([row[-1] for row in group])).get))
        gini = gini_index(groups, groups_value)
        # c45 = -c45_index(groups, groups_value)

        return {'c_label': c, 'value': groups_value, 'group': groups, "proportion": proportion, "score": gini}

    def build_tree_stump(self, dataset, seed):

        final_root = None
        min_gini = np.inf
        best_k = None
        for z in range(dataset.shape[1]-1):
            plus = []
            for j in [i for i in range(dataset.shape[1]-1) if i not in seed]:
                for k in [3, 5, 8, 12, 15]:
                    try:
                        clf = KMeans(n_clusters=k).fit(dataset.iloc[:, seed+[j]])
                        root = self.get_split(clf.labels_, np.array(dataset.iloc[:, seed+[j]+[-1]]))
                        if min_gini > root["score"]:
                            min_gini = root["score"]
                            final_root = root
                            best_k = k
                            plus = [j]
                    except BaseException:
                        pass
            if len(plus) == 0:
                break
            else:
                seed = seed + plus

        return final_root, seed, np.array(dataset.iloc[:, seed]), best_k

    def _predict(self, node, row, c):

        label = []
        error = None
        min_dis = np.inf
        for i, qua in enumerate(c):
            dis = np.sqrt(np.sum(np.square(row - self.train_data[i])))
            if min_dis > dis:
                min_dis = dis
                error = node["c_label"][i]
            if qua == row:
                label.append(node["c_label"][i])
        if len(label) == 0:
            return node["value"][list(set(node["c_label"])).index(error)]

        else:
            return node["value"][list(set(node["c_label"])).index(max(dict(Counter(label)), key=dict(Counter(label)).get))]

    def _predict_log_proba(self, node, row):

        for num, x in enumerate(node["index"]):
            if row <= x:
                return node["proportion"][num]
            elif num == len(node["index"])-1:
                return node["proportion"][num]
            else:
                continue

    def predict(self, x):

        y = []
        if len(self.tree) == 2:
            y = [self.tree["score"] for i in range(len(x))]
        else:
            x = np.array(x)
            clf = KMeans(n_clusters=self.best_n_clusters).fit(np.vstack((x[:, self.feature], self.train_data)))
            for row in clf.labels_[:len(x)]:
                y_pred = self._predict(self.tree, row, clf.labels_[len(x):])
                y.append(y_pred)

        return y


class Decision_trees(BaseEstimator):

    def __init__(self, *, max_depth=10, min_samples_leaf=0.1, criterion="gini", method=None, stump_number=100):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.stump_number = stump_number
        self.evaluate = None
        self.n_feature = 0
        self.n_samples = 0
        self.method = method
        self.criterion = criterion
        self.classes_ = None
        self.tree = []

    def fit(self,  x, y):

        self.classes_ = np.array(sorted(list(dict(Counter(y)).keys())))
        data = pd.DataFrame(x)
        self.n_samples, self.n_feature = data.shape
        self.min_samples_leaf = self.n_samples * self.min_samples_leaf
        data.insert(loc=data.shape[1], column='y', value=y, allow_duplicates=False)
        self.tree = self.build_tree(data, self.max_depth, self.min_samples_leaf)
        file_address = r'print_KMeans.txt'
        f = open(file_address, 'a')
        print(self.tree, file=f)
        f.close()

        return self

    def classification_node(self, data, seed):

        gb = tree_stump().fit(data, seed)

        return gb, gb.tree["score"]

    def build_tree(self, data, max_depth, min_size):  # Build a decision tree

        if self.method == "SMOTE":
            data = np.array(data)
            x_resampled, y_resampled = SMOTE(random_state=0).fit_resample(data[:, :-1], list(map(int, data[:, -1])))
            data = pd.DataFrame(x_resampled)
            data.insert(loc=data.shape[1], column='y', value=y_resampled, allow_duplicates=False)
        elif self.method == "up_sample":
            data = up_sampling(data)
        clf_0, clf_1, clf_2 = self.ova_change(data)
        root = self.get_split(data, clf_0, clf_1, clf_2)
        self.split(root, max_depth, min_size, 1)

        return root

    def ova_change(self, data):

        data_0 = data.copy()
        data_0.loc[data_0.iloc[:, -1] == 2] = 1
        data_1 = data.copy()
        data_1.loc[data_1.iloc[:, -1] == 2] = 0
        data_2 = data.copy()
        data_2.loc[data_2.iloc[:, -1] == 1] = 0
        evaluate_0 = list(Parallel(n_jobs=4)(delayed(self.classification_node)(data_0, seed) for seed in range(self.n_feature)))
        evaluate_1 = list(Parallel(n_jobs=4)(delayed(self.classification_node)(data_1, seed) for seed in range(self.n_feature)))
        evaluate_2 = list(Parallel(n_jobs=4)(delayed(self.classification_node)(data_2, seed) for seed in range(self.n_feature)))
        evaluate_0 = [i[0] for i in sorted(evaluate_0, key=lambda c: c[1])]
        evaluate_1 = [i[0] for i in sorted(evaluate_1, key=lambda c: c[1])]
        evaluate_2 = [i[0] for i in sorted(evaluate_2, key=lambda c: c[1])]

        return evaluate_0[0], evaluate_1[0], evaluate_2[0]

    def get_split(self, data, clf_0, clf_1, clf_2):

        data = np.array(data)
        groups = [[], [], [], []]
        label_0 = clf_0.predict(data)
        label_1 = clf_1.predict(data)
        label_2 = clf_2.predict(data)
        for i in range(data.shape[0]):
            if label_0[i] == 0 and label_1[i] != 1 and label_2[i] != 2:
                groups[0].append(data[i])
            elif label_0[i] != 0 and label_1[i] == 1 and label_2[i] != 2:
                groups[1].append(data[i])
            elif label_0[i] != 0 and label_1[i] != 1 and label_2[i] == 2:
                groups[2].append(data[i])
            else:
                groups[3].append(data[i])
        file_address = r'print_KMeans.txt'
        f = open(file_address, 'a')
        print(len(list(set(clf_0.tree["c_label"]))), len(list(set(clf_1.tree["c_label"]))), len(list(set(clf_2.tree["c_label"]))), file=f)
        print(clf_0.feature, clf_1.feature, clf_2.feature, file=f)
        f.close()

        return {'evaluate': [clf_0, clf_1, clf_2], 'groups': groups, "distribute": [len(i) for i in groups]}


    @staticmethod
    def to_terminal(group):  # Create a terminal node value

        outcomes = [row[-1] for row in group]
        file_address = r'print_KMeans.txt'
        f = open(file_address, 'a')
        print(dict(Counter(outcomes)), file=f)
        f.close()

        return max(dict(Counter(outcomes)), key=dict(Counter(outcomes)).get)

    def split(self, node, max_depth, min_size, depth):

        groups = node['groups']
        del (node['groups'])
        if len(groups[0]) + len(groups[1]) + len(groups[2]) < 5:
            node["value"] = self.to_terminal(groups[3])
        elif len(groups[3]) <= min_size or depth >= max_depth:
            node["value"] = self.to_terminal(groups[3])
        elif len(set([row[-1] for row in groups[3]])) == 1:
            node["value"] = self.to_terminal(groups[3])
        else:
            data = pd.DataFrame(groups[3])
            clf_0, clf_1, clf_2 = self.ova_change(data)
            node['next'] = self.get_split(data, clf_0, clf_1, clf_2)
            print(depth)
            self.split(node['next'], max_depth, min_size, depth + 1)

    def _predict(self, node, row):

        label_0 = node["evaluate"][0].predict(row.reshape(1, -1))
        label_1 = node["evaluate"][1].predict(row.reshape(1, -1))
        label_2 = node["evaluate"][2].predict(row.reshape(1, -1))
        if int(label_0[0]) == 0 and int(label_1[0]) != 1 and int(label_2[0]) != 2:
            return 0
        elif int(label_0[0]) != 0 and int(label_1[0]) == 1 and int(label_2[0]) != 2:
            return 1
        elif int(label_0[0]) != 0 and int(label_1[0]) != 1 and int(label_2[0]) == 2:
            return 2
        else:
            if 'next' in node:
                return self._predict(node['next'], row)
            else:
                return node["value"]

    def predict(self, x):
        x = pd.DataFrame(x)
        x = np.array(x)
        y = []
        for row in x:
            y.append(self._predict(self.tree, row))

        return y

    def predict_proba(self, x):
        x = pd.DataFrame(x)
        x = np.array(x)
        df_empty = []
        for num, row in enumerate(x):
            y_p = [0, 0, 0]
            y_pred_1 = self._predict(self.tree[0], row)
            for key, value in y_pred_1.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_1.values())
            y_pred_2 = self._predict(self.tree[1], row)
            for key, value in y_pred_2.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_2.values())
            y_pred_3 = self._predict(self.tree[2], row)
            for key, value in y_pred_3.items():
                y_p[int(key)] = y_p[int(key)] + value/sum(y_pred_3.values())
            df_empty.append(y_p)

        return np.array(df_empty)

    def gini_index(self, groups, groups_values):  # Calculate the Gini index for a split dataset

        gini = 0.0
        number = 0.0
        for group in groups:
            number += len(group)
        for i, group in enumerate(groups):
            size = sum(row[-1] for row in group)
            if size == 0:
                continue
            else:
                proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
                gini_group = 1 - (proportion ** 2) - ((1.0 - proportion) ** 2)
                gini += gini_group * len(group) / number

        return gini

    def ID3_index(self, groups, groups_values):

        entropy = 0.0
        number = 0.0
        all_group = []
        for group in groups:
            number += len(group)
            all_group += group
        for value in list(set(groups_values)):
            proportion = sum([row[-1] for row in all_group if row[-2] == value]) / float(sum(row[-1] for row in all_group))
            if proportion == 0.0:
                continue
            else:
                entropy -= proportion * np.log2(proportion)
        entropy_groups = 0.0
        for i, group in enumerate(groups):
            size = sum(row[-1] for row in group)
            if size == 0:
                continue
            proportion = sum([row[-1] for row in group if row[-2] == groups_values[i]]) / float(size)
            if proportion == 1.0 or proportion == 0.0:
                entropy_group = 0.0
            else:
                entropy_group = -proportion * np.log2(proportion)-(1-proportion) * np.log2(1-proportion)
            entropy_groups += entropy_group * len(group) / number
        entropy = entropy - entropy_groups

        return entropy


