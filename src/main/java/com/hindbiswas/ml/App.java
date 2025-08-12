package com.hindbiswas.ml;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;

import com.hindbiswas.ml.data.BinaryDataLoader;
import com.hindbiswas.ml.data.Dataset;
import com.hindbiswas.ml.data.MNISTDataLoader;
import com.hindbiswas.ml.models.MultiLayerPerceptron;
import com.hindbiswas.ml.models.Perceptron;
import com.hindbiswas.ml.util.LayerActivations;
import com.hindbiswas.ml.util.LossGradients;
import com.opencsv.exceptions.CsvValidationException;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) {
        try {
            String mlpPath = "/home/shinigami/Documents/image_mlp.json";
            Scanner sc = new Scanner(System.in);
            System.out.println("1. Train a new model");
            System.out.println("2. Use an existing model");

            System.out.print("Choose an option: ");

            int choice = sc.nextInt();
            switch (choice) {
                case 1:
                    mlpTrain(mlpPath);
                    break;
                case 2:
                    mlpUse(mlpPath);
                    break;
                default:
                    System.out.println("Invalid choice");
                    break;
            }

            sc.close();
        } finally {
            ForkJoinPool.commonPool().shutdown(); // cleanup background threads
        }
    }

    public static void mlpTrain(String export) {
        try {
            System.out.println("Loading dataset...");
            Map<String, Dataset> data = MNISTDataLoader.load(
                    "/home/shinigami/Documents/mnist_train.csv", 10000,
                    "/home/shinigami/Documents/mnist_test.csv", 5000);
            Dataset train = data.get("train");
            Dataset test = data.get("test");

            System.out.println("Preparing model...");
            MultiLayerPerceptron mlp = new MultiLayerPerceptron(784, 3, 10);
            mlp.layer(512, LayerActivations.relu());
            mlp.layer(256, LayerActivations.relu());
            mlp.layer(128, LayerActivations.relu());
            mlp.layer(10, LayerActivations.softmax());
            mlp.configure(30, 64, 0.1);
            mlp.lossGradient(LossGradients.softmaxCrossEntropy());

            System.out.println("Training model...");
            mlp.fit(train.X, train.y);

            System.out.println("Evaluating model...");
            System.out.println("Train accuracy: " + mlp.score(train.X, train.y));
            System.out.println("Test  accuracy: " + mlp.score(test.X, test.y));

            mlp.export(Paths.get(export));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void mlpUse(String path) {
        try {
            System.out.println("Loading model...");
            MultiLayerPerceptron mlp = MultiLayerPerceptron.importModel(Paths.get(path));
            Map<String, Dataset> data = MNISTDataLoader.load(
                    "/home/shinigami/Documents/mnist_train.csv", 1,
                    "/home/shinigami/Documents/mnist_test.csv", 10000);
            Dataset test = data.get("test");
            Dataset train = data.get("train");

            System.out.println("Test  accuracy: " + mlp.score(test.X, test.y));
            return;
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
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
