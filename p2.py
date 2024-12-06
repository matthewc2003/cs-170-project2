import time
import math
import numpy as np

class NNClassifier:
    def __init__(self):
        self.training_data = []

    def train(self, training_data):
        self.training_data = training_data

    def test(self, test_instance):
        min_distance = float('inf')
        predicted_label = None

        for features, label in self.training_data:
            distance = math.dist(features, test_instance)
            if distance <= min_distance:
                min_distance = distance
                predicted_label = label

        return predicted_label

class Validator():
    def __init__(self, classifier):
        self.classifier = classifier

    def leave_one_out_validation(self, dataset, feature_subset=None):
        correct_predictions = 0
        total_instances = len(dataset)

        start_time = time.time()
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

            actual_label = test_instance[1]
            is_correct = predicted_label == actual_label
            print(
                f"Iteration {i + 1}/{total_instances}: Predicted = {predicted_label}, "
                f"Actual = {actual_label}, Correct = {is_correct}"
            )

            if is_correct:
                correct_predictions += 1

        end_time = time.time()
        print(correct_predictions, total_instances)
        accuracy = correct_predictions / total_instances
        elapsed_time = end_time - start_time
        print(f"Validation completed in {elapsed_time:.2f} seconds.")
        return accuracy


def load_and_normalize_data(file_path):
    data = []

    with open(file_path, 'r') as file:
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

    with open(file_path, 'r') as file:
        for line in file:
            values = [float(x) for x in line.split()]
            class_label = values[0]
            features = values[1:]
            data.append((features, class_label))

    return data


def main():
    dataset_choice = input("Choose a dataset: Type '1' for small or '2' for large: ").strip()

    if dataset_choice == '1':
        file_path = "small-test-dataset.txt"
    elif dataset_choice == '2':
        file_path = "large-test-dataset.txt"
    else:
        print("Invalid choice. Please type '1' or '2'.")
        return

    print("Loading and normalizing the dataset...")
    dataset = load_and_normalize_data(file_path)
    # dataset = load_data_without_normalization(file_path)

    feature_input = input("Enter the feature subset (space-separated indices, starting from 1): ").strip()
    try:
        feature_subset = [int(f) - 1 for f in feature_input.split()]
    except ValueError:
        print("Invalid input. Please enter space-separated integers.")
        return

    classifier = NNClassifier()
    validator = Validator(classifier)

    print("Performing leave-one-out validation...")
    accuracy = validator.leave_one_out_validation(dataset, feature_subset)

    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()