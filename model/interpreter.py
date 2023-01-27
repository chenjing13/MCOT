import numpy as np
import pandas as pd
from collections import Counter


def local_interpreter(classifier, x, index):

    tree = classifier.tree
    rule = predict(tree, x, index)
    return rule[0]


def global_interpreter(classifier, index, x, y, rules):

    tree = classifier.tree
    for n, row in enumerate(x):
        rule = predict(tree, row, index)
        if rule[1] == y[n]:
            if rule[0] in rules:
                rules[rule[0]] = [rule[1], rules[rule[0]][1] + 1, rules[rule[0]][2] + 1]
            else:
                rules[rule[0]] = [rule[1], 1, 1]
        else:
            if rule[0] in rules:
                rules[rule[0]] = [rule[1], rules[rule[0]][1] + 1, rules[rule[0]][2]]
            else:
                rules[rule[0]] = [rule[1], 1, 0]

    return rules


def global_importance(classifier, index):

    count = {}
    tree = classifier.tree_view()
    for n, t in enumerate(tree):
        whole_tree_feature_importance(t, index, count)
    # print(sorted(count.items(), key=lambda x: x[1], reverse=True))

    return count


def predict(node, row, index, rule=""):

    label_0 = node["evaluate"][0].predict(row.reshape(1, -1))
    label_1 = node["evaluate"][1].predict(row.reshape(1, -1))
    label_2 = node["evaluate"][2].predict(row.reshape(1, -1))
    if rule != "":
        rule = rule + " and "
    if int(label_0[0]) == 0 and int(label_1[0]) != 1 and int(label_2[0]) != 2:
        rule = rule + "feature set {} interacting through hierarchical clustering".format([index[i] for i in node["evaluate"][0].feature])
        rule = rule + " " + "then predict Additive"
        return [rule, 0]
    elif int(label_0[0]) != 0 and int(label_1[0]) == 1 and int(label_2[0]) != 2:
        rule = rule + "feature set {} interacting through hierarchical clustering".format([index[i] for i in node["evaluate"][1].feature])
        rule = rule + " " + "then predict Antagonism"
        return [rule, 1]
    elif int(label_0[0]) != 0 and int(label_1[0]) != 1 and int(label_2[0]) == 2:
        rule = rule + "feature set {} interacting through hierarchical clustering".format([index[i] for i in node["evaluate"][2].feature])
        rule = rule + " " + "then predict Synergy"
        return [rule, 2]
    else:
        if 'next' in node:
            rule = rule + " " + "can not be clearly distinguished by feature set {} for Additive, " \
                                "feature set {} for Antagonism or feature set {} forSynergy".format(
                                 [index[i] for i in node["evaluate"][0].feature], [index[i] for i in node["evaluate"][1].feature], [index[i] for i in node["evaluate"][2].feature])
            return predict(node['next'], row, index, rule)
        else:
            if node["value"] == 0:
                rule = rule + " " + "then predict Additive"
            elif node["value"] == 1:
                rule = rule + " " + "then predict Antagonism"
            else:
                rule = rule + " " + "then predict Antagonism"
            return [rule, node["value"]]


def whole_tree_feature_importance(node, index, count):

    for evaluate in node["evaluate"]:
        for i in evaluate.feature:
            if index[i] in count:
                count[index[i]] = count[index[i]] + 1
            else:
                count[index[i]] = 1

    if "index" in node:
        whole_tree_feature_importance(node['next'], index, count)
    else:
        return count
