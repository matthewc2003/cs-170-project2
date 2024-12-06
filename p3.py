from typing import List
import math
import numpy as np

"""
Group: Matthew Chung - mchun078
DatasetID: 211
Small Dataset Results:
    - Forward: Feature subset: {3, 5}, Acc: 0.92
    - Backward: Feature subset: {3, 5}, Acc: 0.92
Large Dataset Results:
    - Forward: Feature subset: {1, 27}, Acc: 0.955
    - Backward: Feature subset: {27}, Acc: 0.847
Titanic Dataset Results:
    - Forward: Feature subset: {1, 2, 6}, Acc: 0.783
    - Backward: Feature subset: {1, 2, 3, 4}, Acc: 0.794
"""

HEADER = (
    "Welcome to 862327675 Feature Selection Algorithm.\n\n"
    "Type in the name of the file to test: "
)
ALGORITHM_SELECT = "\nType the number of the algorithm you want to run.\n"


class NNClassifier:
    def __init__(self):
        self.training_data = []

    def train(self, training_data):
        self.training_data = training_data

    def test(self, test_instance):
        min_distance = float("inf")
        predicted_label = None

        for features, label in self.training_data:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, test_instance)))
            if distance <= min_distance:
                min_distance = distance
                predicted_label = label

        return predicted_label


class Validator:
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_validation(self, dataset, feature_subset=None):
        correct_predictions = 0
        total_instances = len(dataset)

        for i in range(total_instances):
            test_instance = dataset[i]
            training_data = [dataset[j] for j in range(total_instances) if j != i]

            if feature_subset:
                filtered_training_data = [
                    ([features[k] for k in feature_subset], label) for features, label in training_data
                ]
                filtered_test_features = [test_instance[0][k] for k in feature_subset]
            else:
                filtered_training_data = training_data
                filtered_test_features = test_instance[0]

            self.classifier.train(filtered_training_data)
            predicted_label = self.classifier.test(filtered_test_features)

            if predicted_label == test_instance[1]:
                correct_predictions += 1

        accuracy = correct_predictions / total_instances
        return accuracy


class Algorithm:
    DISPLAY_NAME = None

    def __init__(self, features: int, validator, dataset):
        self.features: int = features
        self.validator = validator
        self.dataset = dataset

    def evaluate(self, state: List[int]) -> float:
        return self.validator.leave_one_out_validation(self.dataset, state)

    def get_initial_state(self) -> List[int]:
        raise NotImplementedError

    def get_options(self, state: List[int]) -> List[List[int]]:
        raise NotImplementedError


class ForwardSelection(Algorithm):
    DISPLAY_NAME = "Forward Selection"

    def get_initial_state(self) -> List[int]:
        return list()

    def get_options(self, state: List[int]) -> List[List[int]]:
        return [
            [feature] + state
            for feature in range(self.features)
            if feature not in state
        ]


class BackwardElimination(Algorithm):
    DISPLAY_NAME = "Backward Elimination"

    def get_initial_state(self) -> List[int]:
        return list(range(self.features))

    def get_options(self, state: List[int]) -> List[List[int]]:
        return [[f for f in state if f != feature] for feature in state]


def load_and_normalize_data(file_path):
    data = []

    with open(file_path, "r") as file:
        for line in file:
            values = [float(x) for x in line.split()]
            class_label = values[0]
            features = values[1:]
            data.append((features, class_label))

    features_matrix = np.array([item[0] for item in data])
    labels = [item[1] for item in data]

    min_val = features_matrix.min(axis=0)
    max_val = features_matrix.max(axis=0)
    normalized_features = (features_matrix - min_val) / (max_val - min_val)

    normalized_data = [(list(normalized_features[i]), labels[i]) for i in range(len(labels))]
    return normalized_data

def load_data_without_normalization(file_path):
    data = []

    with open(file_path, "r") as file:
        for line in file:
            values = [float(x) for x in line.split()]
            class_label = values[0]
            features = values[1:]
            data.append((features, class_label))

    return data


def main():
    file_path = str(input(HEADER))

    print("Loading and normalizing the dataset...")
    # dataset = load_data_without_normalization(file_path)
    dataset = load_and_normalize_data(file_path)
    features = len(dataset[0][0])

    print(f"The dataset has {features} features.")
    classifier = NNClassifier()
    validator = Validator(classifier)

    algorithms = [ForwardSelection, BackwardElimination]
    value = input(
        ALGORITHM_SELECT
        + "".join(f"{count}. {cls.DISPLAY_NAME}\n" for count, cls in enumerate(algorithms, start=1))
    )
    print()

    algorithm = algorithms[int(value) - 1](features, validator, dataset)
    options = [algorithm.get_initial_state()]
    previous_accuracy, best_accuracy, best_state = -1, -1, options[0]

    while True:
        if not options:
            print(
                "Finished search!! "
                f"The best feature subset is {[i + 1 for i in best_state]}, "
                f"which has an accuracy of {best_accuracy * 100:.1f}%"
            )
            break

        evals = [algorithm.evaluate(s) for s in options]

        if len(options) == 1 and not options[0]:
            index = 0
            print(f"Using no features and \"validation\", I get an accuracy of {evals[0] * 100:.1f}%")

        else:
            index = max(range(len(options)), key=lambda x: evals[x])

            for s, e in zip(options, evals):
                print(f"    Using feature(s) {[i + 1 for i in s]} accuracy is {e * 100:.1f}%")

            if len(options) > 1:
                print(f"\nFeature set {[i + 1 for i in options[index]]} was best, accuracy is {evals[index] * 100:.1f}%")

        if previous_accuracy != -1 and evals[index] < previous_accuracy:
            print("(Warning, Accuracy has decreased!)")

        print()

        if evals[index] > best_accuracy:
            best_accuracy, best_state = evals[index], options[index]

        previous_accuracy = evals[index]
        options = algorithm.get_options(sorted(options[index]))


if __name__ == "__main__":
    main()
