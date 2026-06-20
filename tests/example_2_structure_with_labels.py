from neural_network_draw import DrawNN


def run():
    dnn = DrawNN(
        [4, 6, 6, 3],
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        title="Iris Classifier Architecture",
    )

    dnn.draw()


if __name__ == "__main__":
    run()