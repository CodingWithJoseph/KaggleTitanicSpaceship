import math
import numpy as np
import pandas as pd
from collections import defaultdict


# Wrapper class for XGBoost with sigmoid activation and binary cross-entropy loss
class XGBoostSigmoid:
    def __init__(self, params, threshold=0.5, seed=42):
        self.params = params
        self.threshold = threshold  # Threshold to classify sigmoid output as 0 or 1
        self.objective = SigmoidBinaryCrossEntropyObjective()
        self.base = XGBoost(self.params, self.objective, seed)

    def train(self, X, y, num_rounds):
        self.base.fit(X, y, num_rounds)  # Train the underlying boosted trees

    def predict(self, X, as_labels=False, threshold=0.5):
        logits = self.base.predict(X)  # Get raw scores
        probs = self.objective.sigmoid(logits)  # Apply sigmoid to get probabilities
        if as_labels:
            return (probs >= threshold).astype(int)
        return probs

    def score(self, X, y_true):
        raw = self.predict(X)  # Predict probabilities
        predictions = (raw >= self.threshold).astype(int)  # Convert to binary
        return np.mean(predictions == y_true)  # Accuracy


# Core XGBoost class
class XGBoost:
    def __init__(self, params, objective, seed=42):
        self.trees = []  # Store all trained trees
        self.params = defaultdict(lambda: None, params)  # Default values for missing params
        self.objective = objective  # Loss function
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 1e-1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=seed)  # Random number generator

    def fit(self, X, y, num_rounds, verbose=True):
        predictions = self.base_score * np.ones(shape=y.shape)  # Initialize predictions

        for rnd in range(num_rounds):
            gradients = self.objective.gradients(y, predictions)  # Compute gradients
            hessians = self.objective.hessians(y, predictions)  # Compute hessians

            # Row sampling
            idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(y),
                size=math.floor(self.subsample * len(y)),
                replace=False
            )

            # Train one tree on the current gradients
            tree = BoostedTree(
                X=X,
                gradients=gradients,
                hessians=hessians,
                params=self.params,
                max_depth=self.max_depth,
                idxs=idxs
            )
            self.trees.append(tree)

            predictions += self.learning_rate * tree.predict(X)  # Update predictions

            if verbose:
                print(f'Boost Round: [{rnd}] ----> Train Loss = {self.objective.loss(y, predictions)}')

    def predict(self, X):
        # Add predictions from all trees
        return self.base_score + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)


# A single tree in the boosted ensemble
class BoostedTree:
    def __init__(self, X, gradients, hessians, params, max_depth, idxs=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians
        self.params = params
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0
        self._lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0
        self.gamma = self.params['gamma'] if self.params['gamma'] else 0.0
        self.column_subsample = self.params['column_subsample'] if self.params['column_subsample'] else 1.0
        self.max_depth = max_depth
        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))
        self.num_examples = len(self.ridxs)  # Number of training examples
        self.num_features = X.shape[1]  # Number of features
        self.weight = -self.gradients[self.ridxs].sum() / (
                self.hessians[self.ridxs].sum() + self._lambda)  # Leaf weight
        self.split_score = 0.0  # Best gain so far
        self.split_idx = 0  # Feature index for split
        self.threshold = 0.0  # Threshold value for split
        self._build_tree_structure()  # Recursively build the tree

    def _build_tree_structure(self):
        if self.max_depth <= 0:
            return  # Reached max depth, stop recursion

        for fidx in range(self.num_features):
            self._find_best_split_score(fidx)  # Try splitting on each feature

        if self._is_leaf:
            return  # No valid split found, stop here

        feature = self.X[self.ridxs, self.split_idx]
        left_idxs = np.nonzero(feature <= self.threshold)[0]
        right_idxs = np.nonzero(feature > self.threshold)[0]

        # Recursively build left and right subtrees
        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                self.max_depth - 1, self.ridxs[left_idxs])
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                 self.max_depth - 1, self.ridxs[right_idxs])

    def _find_best_split_score(self, fidx):
        feature = self.X[self.ridxs, fidx]
        gradients = self.gradients[self.ridxs]
        hessians = self.hessians[self.ridxs]

        sorted_idxs = np.argsort(feature)
        sorted_feature = feature[sorted_idxs]
        sorted_gradient = gradients[sorted_idxs]
        sorted_hessians = hessians[sorted_idxs]

        hessian_sum = sorted_hessians.sum()
        gradient_sum = sorted_gradient.sum()

        right_hessian_sum = hessian_sum
        right_gradient_sum = gradient_sum
        left_hessian_sum = 0.0
        left_gradient_sum = 0.0

        for idx in range(0, self.num_examples - 1):
            candidate = sorted_feature[idx]
            neighbor = sorted_feature[idx + 1]

            gradient = sorted_gradient[idx]
            hessian = sorted_hessians[idx]

            right_gradient_sum -= gradient
            right_hessian_sum -= hessian
            left_gradient_sum += gradient
            left_hessian_sum += hessian

            if right_hessian_sum <= self.min_child_weight:
                return  # Stop if the right child is too small

            # Compute gain from potential split
            right_score = (right_gradient_sum ** 2) / (right_hessian_sum + self._lambda)
            left_score = (left_gradient_sum ** 2) / (left_hessian_sum + self._lambda)
            score_before_split = (gradient_sum ** 2) / (hessian_sum + self._lambda)
            gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma

            # Save split if it's the best so far
            if gain > self.split_score:
                self.split_score = gain
                self.split_idx = fidx
                self.threshold = (candidate + neighbor) / 2

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict_row(example) for example in X])  # Predict each row

    def _predict_row(self, example):
        if self._is_leaf:
            return self.weight  # Return leaf weight
        child = self.left if example[self.split_idx] <= self.threshold else self.right
        return child._predict_row(example)  # Recurse down the tree

    @property
    def _is_leaf(self):
        return self.split_score == 0.0  # Leaf node if no gain found


# Sigmoid binary cross-entropy loss used for classification
class SigmoidBinaryCrossEntropyObjective:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # Standard sigmoid

    @staticmethod
    def loss(labels, predictions):
        probs = SigmoidBinaryCrossEntropyObjective.sigmoid(predictions)
        epsilon = 1e-15  # To avoid log(0)
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))  # Binary log loss

    @staticmethod
    def gradients(labels, predictions):
        probs = SigmoidBinaryCrossEntropyObjective.sigmoid(predictions)
        return probs - labels  # Gradient of binary cross-entropy

    @staticmethod
    def hessians(labels, predictions):
        probs = SigmoidBinaryCrossEntropyObjective.sigmoid(predictions)
        return probs * (1 - probs)  # Hessian for sigmoid cross-entropy