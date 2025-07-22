package com.hindbiswas.ml;

import java.io.IOException;
import java.util.Map;

import com.hindbiswas.ml.data.BinaryDataLoader;
import com.hindbiswas.ml.data.Dataset;
import com.hindbiswas.ml.models.Perceptron;
import com.opencsv.exceptions.CsvValidationException;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) {
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
