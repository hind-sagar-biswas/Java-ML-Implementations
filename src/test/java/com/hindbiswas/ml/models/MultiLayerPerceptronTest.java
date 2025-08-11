package com.hindbiswas.ml.models;

import org.junit.jupiter.api.Test;

import com.hindbiswas.ml.util.LayerActivations;
import com.hindbiswas.ml.util.Matrix;

import static org.junit.jupiter.api.Assertions.*;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Random;

public class MultiLayerPerceptronTest {

    /**
     * Train a small MLP on a simple linearly-separable 2D dataset.
     * Expect the model to reach high accuracy after training.
     *
     * If this test is flaky on your machine, increase epochs or the hidden layer
     * size,
     * or set a deterministic seed for your weight init (see Matrix.xavier).
     */
    @Test
    public void testTrainBinaryClassifier() throws Exception {
        final int inputSize = 2;
        final int hiddenLayers = 1;
        final int outputSize = 2;

        // Create MLP with a slightly larger learning rate (constructor: inputSize,
        // hiddenLayers, outputSize, learningRate)
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(inputSize, hiddenLayers, outputSize, 0.05);

        // Add layers: 4-unit hidden (sigmoid), 2-unit softmax output
        mlp.layer(4, LayerActivations.sigmoid());
        mlp.layer(2, LayerActivations.softmax());

        // Configure: epochs, batchSize (not used in your impl yet), validationSplit
        mlp.configure(1500, 1, 0.0);

        // Use softmax + cross-entropy gradient: pred - label (dL/dz)
        mlp.loss((pred, label) -> pred.minus(label));

        // Build a small synthetic dataset (linearly separable):
        // label = 1 if x0 + x1 > 1.0 else 0
        int N = 200;
        ArrayList<ArrayList<Double>> X = new ArrayList<>();
        ArrayList<Double> Y = new ArrayList<>();
        Random rnd = new Random(42); // deterministic features

        for (int i = 0; i < N; i++) {
            double x0 = rnd.nextDouble();
            double x1 = rnd.nextDouble();
            ArrayList<Double> features = new ArrayList<>();
            features.add(x0);
            features.add(x1);
            X.add(features);
            int label = (x0 + x1 > 1.0) ? 1 : 0;
            Y.add((double) label);
        }

        // Fit the model (this should set fitted = true)
        mlp.fit(X, Y);

        // Evaluate accuracy on the same dataset (small unit test, so reuse training
        // set)
        int correct = 0;
        for (int i = 0; i < N; i++) {
            SimpleMatrix input = Matrix.columnWithoutBias(X.get(i)); // inputSize x 1
            SimpleMatrix out = mlp.predict(input); // outputSize x 1
            int predicted = argMax(out);
            int actual = Y.get(i).intValue();
            if (predicted == actual)
                correct++;
        }

        double accuracy = correct / (double) N;
        // Expect quite high accuracy on a simple separable dataset
        assertTrue(accuracy >= 0.90, () -> "Expected accuracy >= 0.90 but was " + accuracy);
    }

    // Helper: returns index of max element in an (n x 1) SimpleMatrix
    private int argMax(SimpleMatrix v) {
        int best = 0;
        double bestVal = v.get(0, 0);
        for (int i = 1; i < v.getNumRows(); i++) {
            double val = v.get(i, 0);
            if (val > bestVal) {
                bestVal = val;
                best = i;
            }
        }
        return best;
    }
}
