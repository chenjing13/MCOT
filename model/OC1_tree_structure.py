#
#
#.......Defining the tree structure of Oblique Clasiifier 1 (OC1) for classification problems...................
#
#
#
#....Importing all the packages..........
#
#
import numpy as np
from collections import Counter
#
#
#
#.....Defining the tree structure................
#
#
class Tree:
    def __init__(self, n_features, is_classifier): # Defining parameters for a single node.
        self.n_features = n_features               # 'n_features' is the number of features used to define the node.
        self.is_classifier = is_classifier         # 'True' for classification problem.
        self.root_node = None                      #  Initially 'root_node' will be set to 'None'.
        self.depth = -1                            #  Depth of the node.
        self.num_leaf_nodes = -1                   #  Number of leaf nodes in the internal nodes.

    def set_root_node(self, root_node):
        self.root_node = root_node

    def set_depth(self, depth):
        self.depth = depth

    def get_depth(self):
        if self.depth == -1:
            NotImplementedError('TODO: depth first traversal')
        return self.depth

    def predict(self, X):
        return self.root_node.predict(X)

    def predict_proba(self, X):
        return self.root_node.predict_proba(X)

class Node:
    def __init__(self, w, b, is_classifier=True, value=None, conf=0.0, samples=None, features=None):
        self.w = w                  # weights
        self.b = b                  # bias
        self.value = value          # value of the current node if the node is a leaf
        self.conf = conf            # score of the node, typically accuracy or r2
        self.samples = samples      # training examples in this node
        self.features = features    # features used in this node
        self.is_classifier = is_classifier
        self.left_child = None
        self.right_child = None
        self.is_fitted = False      # flag to check if the node has been fitted

    def add_left_child(self, child):
        self.left_child = child

    def add_right_child(self, child):
        self.right_child = child

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def predict(self, X):

        # Partition the data based on the split
        y = (np.dot(X, self.w) + self.b).squeeze()
        left, right = (y <= 0), (y > 0)
        y[left] = self.left_child.predict(X[left, :])
        y[right] = self.right_child.predict(X[right, :])

        return y

    def predict_proba(self, X):

        y = (np.dot(X, self.w) + self.b).squeeze()
        left, right = (y <= 0), (y > 0)
        y_p = []
        y_p_left = self.left_child.predict_proba(X[left, :])
        y_p_right = self.right_child.predict_proba(X[right, :])
        l, r = 0, 0
        if X.shape[0] != 0:
            for i in range(X.shape[0]):
                if not isinstance(left, np.bool_):
                    if left[i]:
                        y_p.append(y_p_left[l])
                        l = l + 1
                    else:
                        y_p.append(y_p_right[r])
                        r = r + 1
                else:
                    if left:
                        y_p.append(y_p_left[0])
                    else:
                        y_p.append(y_p_right[0])

        return np.array(y_p)


class LeafNode(Node):
    def __init__(self, is_classifier=True, value=None, conf=0.0, samples=None, features=None, y=None):

        self.y = y

        super(LeafNode, self).__init__(w=None, b=None, is_classifier=is_classifier,
                                       value=value, conf=conf, samples=samples, features=features)

    def predict(self, X):

        # Simply return the leaf value
        return np.full((X.shape[0], ), self.value)

    def predict_proba(self, X):

        y_proba = []
        for i in range(X.shape[0]):
            count = Counter(self.y)
            proba = [0, 0, 0]
            proba[0] = count[0]/len(self.y)
            proba[1] = count[1]/len(self.y)
            proba[2] = count[2]/len(self.y)
            y_proba.append(proba)

        return np.array(y_proba)