import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label
    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)
def gini_impurity(class_vector):
    true = 0
    false = 0
    for item in class_vector:
        if item == 1:
            true += 1
        else:
            false += 1
    true_prob = true/len(class_vector)
    false_prob = false/len(class_vector)
    gini_impurity_p = 1 - true_prob**2 - false_prob**2
    return gini_impurity_p
def gini_gain(previous_classes, current_classes):
    total_length = sum(len(row) for row in current_classes)
    gini = 0
    for item in current_classes:
        length = len(item)
        gini += (length/total_length) * gini_impurity(item)
    gini_gain = gini_impurity(previous_classes) - gini
    return gini_gain
class DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit
    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        counter = Counter(classes)
        if len(counter) == 0:
            return DecisionNode(None, None, None, class_label = 0)
        if len(counter) == 1:
            return DecisionNode(None, None, None, class_label = list(counter.keys())[0])
        if depth == self.depth_limit:
            if list(counter.values())[0] > list(counter.values())[1]:
                return DecisionNode(None, None, None, class_label = 0)
            else:
                return DecisionNode(None, None, None, class_label = 1)       
        mean = np.mean(features, axis=0)
        best_alpha = 0
        best_gain = float('-inf')
        for alpha in range(np.size(features,axis=1)):
            left = classes[features[:,alpha] < mean[alpha]]
            right= classes[features[:,alpha] >= mean[alpha]]
            if np.size(left)!=0 and np.size(right)!=0:
                gain=gini_gain(classes,[left, right])
            else:
                gain=0 
            if best_gain < gain:
                best_gain = gain
                best_alpha = alpha
        features_l = features[features[:,best_alpha] < mean[best_alpha]]
        features_r = features[features[:,best_alpha] >= mean[best_alpha]]
        classes_l = classes[features[:,best_alpha] < mean[best_alpha]]
        classes_r = classes[features[:,best_alpha] >= mean[best_alpha]]
        left_node = self.__build_tree__(features_l,classes_l,depth = depth + 1)
        right_node = self.__build_tree__(features_r,classes_r,depth = depth + 1)
        root = DecisionNode(left_node,right_node,lambda features:features[best_alpha] < mean[best_alpha])
        return root
    def classify(self, features):
        class_labels = []
        class_labels = [self.root.decide(example) for example in features]
        return class_labels
class RandomForest:

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        self.trees = []
        self.attr_list = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
    def fit(self, features, classes):
        shuffler1 = list(range(np.size(features,axis=0)))
        shuffler2 = list(range(np.size(features,axis=1)))
        
        num_examples = np.size(features,axis=0)
        num_features = np.size(features,axis=1)
        for i in range(self.num_trees):
            np.random.shuffle(shuffler1)
            np.random.shuffle(shuffler2)
            indices_1 = shuffler1[0: int(self.example_subsample_rate * num_examples)]
            indices_2 = shuffler2[0: int(self.attr_subsample_rate * num_features)]
            self.attr_list.append(indices_2)
            selected_features = features[indices_1]
            selected_features = selected_features[:, indices_2]
            selected_classes = classes[indices_1]
            tree=DecisionTree(self.depth_limit)
            tree.fit(selected_features,selected_classes)
            self.trees.append(tree)
    def classify(self, features):
        # TODO: finish this.
        class_labels = []
        classes = []
        for i in range(self.num_trees):
            classes.append([])
        cutoff = self.num_trees / 2
        for example in features:
            vote = 0
            idx = 0
            for tree in self.trees:
                temp = example[self.attr_list[idx]]
                a = tree.root.decide(temp)
                vote += a
                classes[idx].append(a)
                idx += 1
            if vote > cutoff:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return class_labels, classes
class ChallengeClassifier:
    def __init__(self):
        self.random_forest=RandomForest(20,8,0.7,0.7)

    def fit(self, features, classes):

        self.random_forest.fit(features,classes)

    def classify(self, features):

        return self.random_forest.classify(features)


