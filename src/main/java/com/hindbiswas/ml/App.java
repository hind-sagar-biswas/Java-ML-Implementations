package com.hindbiswas.ml;

import java.io.IOException;
import java.util.Map;

import com.hindbiswas.ml.data.BinaryDataLoader;
import com.hindbiswas.ml.data.Dataset;
import com.hindbiswas.ml.models.MultiLayerPerceptron;
import com.hindbiswas.ml.models.Perceptron;
import com.hindbiswas.ml.util.Activations;
import com.opencsv.exceptions.CsvValidationException;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(784, 2, 10);
        mlp.layer(256, Activations.sigmoid());
        mlp.layer(128, Activations.sigmoid());
        mlp.layer(10, Activations.softmax());
        mlp.configure(10, 2000, 0.2);
    }

    public static void perceptron() {
        String csv = "/home/shinigami/Downloads/iris.csv";

        Map<String, Dataset> data;

        try {
            data = BinaryDataLoader.loadAndSplit(csv, 42L, "setosa", "versicolor");
        } catch (CsvValidationException | IOException e) {
            e.printStackTrace();
            return;
        }

        Dataset train = data.get("train");
        Dataset test = data.get("test");

        Perceptron model = new Perceptron().shuffle(true).verbose(true);

        model.fit(train.X, train.y);

        System.out.println(model);
        System.out.println("Train accuracy: " + model.score(train.X, train.y));
        System.out.println("Test  accuracy: " + model.score(test.X, test.y));

        for (int i = 0; i < test.X.size(); i++) {
            int predicted = model.predict(test.X.get(i));
            int actual = test.y.get(i).intValue();

            System.out.printf("Sample %d: Predicted=%d, Actual=%d%n", i, predicted, actual);
        }
    }
}
