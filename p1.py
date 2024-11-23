from typing import List
import random


HEADER = (
    "Welcome to 862327675 Feature Selection Algorithm.\n\n"
    "Please enter total number of features: "
)
ALGORITHM_SELECT = "\nType the number of the algorithm you want to run.\n"


class Algorithm:
    DISPLAY_NAME = None

    def __init__(self, features: int):
        self.features: int = features

    @staticmethod
    def evaluate(state: List[int]) -> float:
        return random.random()
    
    def get_initial_state(self) -> List[int]:
        raise NotImplementedError

    def get_options(self, state: List[int]) -> List[List[int]]:
        return NotImplementedError

    # performs one step & returns next state
    def search(self, state: List[int]) -> bool:
        raise NotImplementedError


class ForwardSelection(Algorithm):
    DISPLAY_NAME = "Forward Selection"

    def get_initial_state(self) -> List[int]:
        return list()

    def get_options(self, state: List[int]) -> List[List[int]]:
        return [
            [feature] + state
            for feature in range(1, self.features + 1)
            if feature not in state
        ]


class BackwardElimination(Algorithm):
    DISPLAY_NAME = "Backward Elimination"

    def get_initial_state(self) -> List[int]:
        return list(range(1, self.features + 1))

    def get_options(self, state: List[int]) -> List[List[int]]:
        return [
            [f for f in state if f != feature]
            for feature in state
        ]


def main():
    features = int(input(HEADER))

    algorithms = [ForwardSelection, BackwardElimination]
    value = input(
        ALGORITHM_SELECT +
        "".join(f"{count}. {cls.DISPLAY_NAME}\n" for count, cls in enumerate(algorithms, start=1))
    )
    print()

    algorithm = algorithms[int(value) - 1](features)
    options = [algorithm.get_initial_state()]
    previous_accuracy, best_accuracy, best_state = -1, -1, options[0]

    while True:
        if not options:
            print(
                "Finished search!! "
                f"The best feature subset is {best_state}, "
                f"which has an accuracy of {best_accuracy * 100:.1f}%"
            )
            break

        evals = [algorithm.evaluate(s) for s in options]

        if len(options) == 1 and not options[0]:
            index = 0
            print(f"Using no features and \"random\" evaluation, I get an accuracy of {evals[0] * 100:.1f}%")
        
        else:
            index = max(range(len(options)), key=lambda x: evals[x])
            
            for s, e in zip(options, evals):
                print(f"    Using feature(s) {s} accuracy is {e * 100:.1f}%")
            
            if len(options) > 1:
                print(f"\nFeature set {options[index]} was best, accuracy is {evals[index] * 100:.1f}%")

        if previous_accuracy != -1 and evals[index] < previous_accuracy:
            print("(Warning, Accuracy has decreased!)")

        print()

        if evals[index] > best_accuracy:
            best_accuracy, best_state = evals[index], options[index]

        previous_accuracy = evals[index]
        options = algorithm.get_options(sorted(options[index]))


if __name__ == "__main__":
    main()