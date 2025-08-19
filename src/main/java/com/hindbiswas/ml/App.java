package com.hindbiswas.ml;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;

import com.hindbiswas.ml.data.BinaryDataLoader;
import com.hindbiswas.ml.data.DataFrame;
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
        perceptron();
    }

    public static void mlp() {
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
            Map<String, DataFrame> data = MNISTDataLoader.load(
                    "/home/shinigami/Documents/mnist_train.csv", 10000,
                    "/home/shinigami/Documents/mnist_test.csv", 5000);
            DataFrame train = data.get("train");
            DataFrame test = data.get("test");

            System.out.println("Preparing model...");
            MultiLayerPerceptron mlp = new MultiLayerPerceptron(784, 3, 10);
            mlp.layer(512, LayerActivations.relu());
            mlp.layer(256, LayerActivations.relu());
            mlp.layer(128, LayerActivations.relu());
            mlp.layer(10, LayerActivations.softmax());
            mlp.configure(30, 64, 0.1);
            mlp.lossGradient(LossGradients.softmaxCrossEntropy());

            System.out.println("Training model...");
            mlp.fit(train);

            System.out.println("Evaluating model...");
            System.out.println("Train accuracy: " + mlp.score(train));
            System.out.println("Test  accuracy: " + mlp.score(test));

            mlp.export(Paths.get(export));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void mlpUse(String path) {
        try {
            System.out.println("Loading model...");
            MultiLayerPerceptron mlp = MultiLayerPerceptron.importModel(Paths.get(path));
            Map<String, DataFrame> data = MNISTDataLoader.load(
                    "/home/shinigami/Documents/mnist_train.csv", 1,
                    "/home/shinigami/Documents/mnist_test.csv", 10000);
            DataFrame test = data.get("test");
            DataFrame train = data.get("train");

            System.out.println("Test  accuracy: " + mlp.score(test));
            return;
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
    }

    public static void perceptron() {
        String csv = "/home/shinigami/Downloads/iris.csv";

        Map<String, DataFrame> data;

        try {
            data = BinaryDataLoader.loadAndSplit(csv, 42L, "setosa", "versicolor");
        } catch (CsvValidationException | IOException e) {
            e.printStackTrace();
            return;
        }

        DataFrame train = data.get("train");
        DataFrame test = data.get("test");

        System.out.println("================================");
        System.out.println("Training Dataset");
        System.out.println("--------------------------------");
        System.out.println(train);
        System.out.println("Shape: " + train.shape()[0] + "x" + train.shape()[1]);
        System.out.println(train.summary());
        System.out.println("================================");
        System.out.println("Testing Dataset");
        System.out.println("--------------------------------");
        System.out.println(test);
        System.out.println("Shape: " + test.shape()[0] + "x" + test.shape()[1]);
        System.out.println(test.summary());
        System.out.println("================================");

        Perceptron model = new Perceptron().shuffle(true).verbose(true);

        model.fit(train);

        System.out.println(model);
        System.out.println("Train accuracy: " + model.score(train));
        System.out.println("Test  accuracy: " + model.score(test));
    }
}
