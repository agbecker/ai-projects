# Tutorial from https://www.youtube.com/watch?v=sgQAhG5Q7iY

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classes
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        
        # For decision node
        self.feature_index = feature_index  # Feature evaluated for the decision
        self.threshold = threshold          # Value the feature is compared to
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.info_gain = info_gain          # Information gain from the decision

        # For leaf node
        self.value = value                  # Majority class of the leaf

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2, impurity_test='gini'):
        
        # Initialize tree root
        self.root = None

        # Stopping conditions
        self.min_samples_split = min_samples_split  # Stop building tree if node has fewer samples than this
        self.max_depth = max_depth                  # Stop building tree if this depth has been reached

        # Define impurity test
        self.impurity_test = impurity_test

    def build_tree(self, dataset, curr_depth=0):
        """ Recursive function to build the tree """

        # Organize data
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # If stopping hasn't been met, this will be a decision node
        # Recursively add nodes until stopping condition is met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # Find the best split
            best_split = self.get_best_split(dataset, num_features)

            # Check if information gain is positive
            if best_split['info_gain'] > 0:
                # Expand left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                # Expand right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)

                # Return decision node
                return Node(best_split['feature_index'], best_split['threshold'],
                            left_subtree, right_subtree, best_split['info_gain'])
        
        # If cannot expand further, return leaf node
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_features):
        """ Function to find the comparison with the greatest information gain """

        # Result dictionary
        best_split = dict()
        max_info_gain = -float('inf')

        # Loop over all features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values) # Obtains all possible values to compare

            # Loop over all the values of that feature
            for threshold in possible_thresholds:
                # Get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                # Check if children are not null
                # An empty child means no separation was performed (worst possible scenario)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    # Compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)

                    # Update the best split if improved
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        # Return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        """ Splits the dataset into left and right, comparing the feature to the threshold """

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child):
        """ Returns information gain from the split """

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if self.impurity_test == 'gini':
            impurity = self.gini_index
        else:
            impurity = self.entropy

        gain = impurity(parent) - (weight_l*impurity(l_child) + weight_r*impurity(r_child))
        return gain
    
    def entropy(self, y):
        """ Computes entropy """

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)

        return entropy
    

    def gini_index(self, y):
        """ Computes Gini index """

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2

        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        """ Returns the class chosen by the leaf """

        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f'X_{tree.feature_index} <= {tree.threshold} ? {tree.info_gain}')
            print(f'{indent}left:', end='')
            self.print_tree(tree.left, indent+indent)
            print(f'{indent}right:', end='')
            self.print_tree(tree.right, indent+indent)

    def fit(self, X, Y):
        """ Trains the tree """

        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """ Inference on test dataset """

        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        """ Classifies a single instance """

        # Returns value if node is leaf
        if tree.value is not None:
            return tree.value
        
        # Else compares and descends
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
# ----------------------------------------------------------------

if __name__ == '__main__':
    # Setting up data
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
    data = pd.read_csv('./iris.csv', skiprows=1, header=None, names=col_names)

    # Train-test split
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

    # Training the tree
    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train, Y_train)
    classifier.print_tree()

    # Testing the tree
    Y_pred = classifier.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))