"""
Usage examples for neural_network_draw.

Run with:
    python3 test_script.py

Each example saves a PNG. If you're running this locally with a normal
display, you can remove the `matplotlib.use("Agg")` line and the
`plt.savefig(...)` calls -- `network.draw()` / `dnn.draw()` will just pop
up a window instead.
"""

import matplotlib
matplotlib.use("Agg")  # remove this line for an interactive window
import matplotlib.pyplot as plt

from neural_network_draw import NeuralNetwork, DrawNN


def save(name):
    plt.savefig(f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {name}.png")


# ---------------------------------------------------------------------------
# Example 1: Old-style API (backward compatibility check)
# Anyone using the package before this upgrade should see no breaking changes.
# ---------------------------------------------------------------------------
def example_1_basic_structure():
    network = NeuralNetwork(12)
    network.add_layer(4)
    network.add_layer(8)
    network.add_layer(6)
    network.add_layer(10)
    network.add_layer(3)
    network.draw()
    save("example_1_basic_structure")


# ---------------------------------------------------------------------------
# Example 2: High-level DrawNN with just structure + labels + title
# Good for a quick architecture diagram with no weights/biases.
# ---------------------------------------------------------------------------
def example_2_structure_with_labels():
    dnn = DrawNN(
        [4, 6, 6, 3],
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        title="Iris Classifier Architecture",
    )
    dnn.draw()
    save("example_2_structure_with_labels")


# ---------------------------------------------------------------------------
# Example 3: Low-level NeuralNetwork API with full feature set
# weights, biases, activation labels, input/output labels, custom title
# ---------------------------------------------------------------------------
def example_3_low_level_full_features():
    network = NeuralNetwork(4, title="Handwritten Digit Classifier (Simplified)")

    network.add_layer(3, labels=["pixel_1", "pixel_2", "pixel_3"])

    network.add_layer(
        4,
        weights=[
            [0.10, -0.20, 0.30, 0.40],
            [0.50, 0.60, -0.70, 0.80],
            [-0.10, 0.20, 0.30, -0.40],
        ],
        biases=[0.10, 0.20, 0.30, 0.40],
        activation="ReLU",
    )

    network.add_layer(
        2,
        weights=[
            [0.20, -0.30],
            [0.10, 0.40],
            [0.50, -0.60],
            [0.20, 0.10],
        ],
        biases=[0.05, -0.05],
        activation="Softmax",
        labels=["Cat", "Dog"],
    )

    network.draw(show_weights=True)
    save("example_3_low_level_full_features")


# ---------------------------------------------------------------------------
# Example 4: High-level DrawNN API, same network as example 3, one call
# ---------------------------------------------------------------------------
def example_4_high_level_full_features():
    dnn = DrawNN(
        [3, 4, 2],
        weights=[
            [
                [0.10, -0.20, 0.30, 0.40],
                [0.50, 0.60, -0.70, 0.80],
                [-0.10, 0.20, 0.30, -0.40],
            ],
            [
                [0.20, -0.30],
                [0.10, 0.40],
                [0.50, -0.60],
                [0.20, 0.10],
            ],
        ],
        biases=[None, [0.10, 0.20, 0.30, 0.40], [0.05, -0.05]],
        activations=[None, "ReLU", "Softmax"],
        input_labels=["x1", "x2", "x3"],
        output_labels=["Cat", "Dog"],
        title="High-Level API Example",
    )
    dnn.draw(show_weights=True)
    save("example_4_high_level_full_features")


# ---------------------------------------------------------------------------
# Example 5: Weights and biases WITHOUT showing the numeric weight values
# Connection color/thickness still encodes sign and magnitude -- useful for
# a cleaner look when you don't need exact numbers visible.
# ---------------------------------------------------------------------------
def example_5_weights_without_numbers():
    dnn = DrawNN(
        [4, 5, 3],
        weights=[
            [[0.8, -0.3, 0.5, 0.1, -0.6],
             [-0.2, 0.7, 0.4, -0.5, 0.3],
             [0.6, 0.1, -0.8, 0.2, 0.5],
             [-0.4, 0.5, 0.3, -0.7, 0.2]],
            [[0.5, -0.4, 0.6],
             [0.2, 0.8, -0.3],
             [-0.6, 0.3, 0.5],
             [0.4, -0.2, 0.7],
             [0.1, 0.5, -0.4]],
        ],
        title="Weight Strength Shown by Color/Thickness Only",
    )
    dnn.draw(show_weights=False)  # weights still color/size the lines
    save("example_5_weights_without_numbers")


# ---------------------------------------------------------------------------
# Example 6: No labels at all, multiple hidden layers, biases only
# Common case: you just want to visualize a trained model's bias values.
# ---------------------------------------------------------------------------
def example_6_biases_only():
    network = NeuralNetwork(6, title="Layer Biases Overview")
    network.add_layer(4)
    network.add_layer(6, biases=[0.12, -0.05, 0.33, -0.21, 0.08, 0.40], activation="ReLU")
    network.add_layer(5, biases=[-0.10, 0.22, 0.15, -0.30, 0.05], activation="ReLU")
    network.add_layer(2, biases=[0.18, -0.18], activation="Sigmoid")
    network.draw()
    save("example_6_biases_only")


# ---------------------------------------------------------------------------
# Example 7: Single hidden layer, minimal network (smallest realistic case)
# ---------------------------------------------------------------------------
def example_7_minimal_network():
    dnn = DrawNN(
        [2, 3, 1],
        input_labels=["x1", "x2"],
        output_labels=["y"],
        title="Minimal Network (XOR-style)",
    )
    dnn.draw()
    save("example_7_minimal_network")


# ---------------------------------------------------------------------------
# Example 8: Many input features (tests the left-side label column with
# a longer, denser list of labels)
# ---------------------------------------------------------------------------
def example_8_many_input_features():
    dnn = DrawNN(
        [8, 10, 4],
        input_labels=[
            "age", "income", "credit_score", "years_employed",
            "loan_amount", "existing_debt", "num_accounts", "region_code",
        ],
        output_labels=["Approve", "Review", "Deny", "Refer"],
        title="Loan Approval Model Architecture",
    )
    dnn.draw()
    save("example_8_many_input_features")


# ---------------------------------------------------------------------------
# Example 9: Regression-style network (single output, no class labels needed)
# ---------------------------------------------------------------------------
def example_9_regression_network():
    dnn = DrawNN(
        [5, 8, 8, 1],
        input_labels=["bedrooms", "bathrooms", "sqft", "lot_size", "year_built"],
        output_labels=["predicted_price"],
        title="House Price Regression Network",
    )
    dnn.draw()
    save("example_9_regression_network")


if __name__ == "__main__":
    example_1_basic_structure()
    example_2_structure_with_labels()
    example_3_low_level_full_features()
    example_4_high_level_full_features()
    example_5_weights_without_numbers()
    example_6_biases_only()
    example_7_minimal_network()
    example_8_many_input_features()
    example_9_regression_network()
    print("\nAll examples ran successfully.")
