package com.hindbiswas.ml.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

/**
 * Perceptron
 */
public class Perceptron {
    private Double learningRate = 0.01;
    private Integer iterations = 1000;
    private Double threshold = 0d;
    private Boolean shuffle = false;
    private Boolean verbose = false;
    private SimpleMatrix theta = null;
    private Integer weightSeed = null;

    private Activation activation = x -> x >= threshold ? 1 : -1;

    public Perceptron() {
    }

    public Perceptron(Double learningRate, Integer iterations, Double threshold, Boolean shuffle, Boolean verbose) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.threshold = threshold;
        this.shuffle = shuffle;
        this.verbose = verbose;
    }

    public Perceptron(Double learningRate, Integer iterations, Double threshold) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.threshold = threshold;
    }

    public Perceptron(Double learningRate, Double threshold) {
        this.learningRate = learningRate;
        this.threshold = threshold;
    }

    public Perceptron(Double threshold) {
        this.threshold = threshold;
    }

    public Perceptron randomizeWeights(int seed) {
        if (theta != null) {
            throw new IllegalStateException("Model has already been fitted. Cannot randomize weights.");
        }
        this.weightSeed = seed;
        return this;
    }

    public Perceptron verbose(Boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    public Perceptron shuffle(Boolean shuffle) {
        if (theta != null) {
            throw new IllegalStateException("Model has already been fitted. Cannot shuffle data.");
        }
        this.shuffle = shuffle;
        return this;
    }

    public Perceptron withActivation(Activation act) {
        this.activation = act;
        return this;
    }

    public Perceptron fit(
            ArrayList<ArrayList<Double>> dataX,
            ArrayList<Double> dataY) throws IllegalArgumentException {
        if (dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }
        if (dataX.get(0).isEmpty()) {
            throw new IllegalArgumentException("Features must be non-empty.");
        }
        for (Double y : dataY) {
            if (y != -1 && y != 1) {
                throw new IllegalArgumentException("DataY values must be either -1 or 1.");
            }
        }

        int m = dataX.size();
        int n = dataX.get(0).size() + 1; // +1 for bias term

        seedWeights(n);

        for (int epoch = 0; epoch < iterations; epoch++) {
            boolean failed = false;

            ArrayList<Integer> indices = new ArrayList<>();
            for (int i = 0; i < m; i++)
                indices.add(i);
            if (shuffle)
                Collections.shuffle(indices);

            for (int idx : indices) {
                double[] xiArr = new double[n];
                xiArr[0] = 1.0;
                for (int j = 0; j < n - 1; j++) {
                    xiArr[j + 1] = dataX.get(idx).get(j);
                }

                SimpleMatrix inputs = new SimpleMatrix(n, 1, true, xiArr);
                double expected = dataY.get(idx);

                // prediction
                double raw = theta.transpose().mult(inputs).get(0, 0);
                int prediction = activation.apply(raw);
                double error = (expected - prediction) * learningRate;
                theta = theta.plus(inputs.scale(error));

                if (prediction != expected) {
                    failed = true;
                }
            }

            if (verbose) {
                System.out.println("Epoch " + (epoch + 1) + "/" + iterations +
                        " — current weights: " + Arrays.toString(theta.getDDRM().getData()));
            }

            if (!failed)
                break;
        }

        return this;
    }

    private void seedWeights(int n) {
        theta = new SimpleMatrix(n, 1); // Initialize theta with zeros

        if (weightSeed != null) {
            Random rand = new Random(weightSeed);
            for (int i = 0; i < n; i++) {
                theta.set(i, 0, rand.nextDouble());
            }
        }
    }

    public Integer predict(ArrayList<Double> x) throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        double[] xArray = new double[x.size() + 1];
        xArray[0] = 1;
        for (int i = 0; i < x.size(); i++) {
            xArray[i + 1] = x.get(i);
        }

        SimpleMatrix xMatrix = new SimpleMatrix(1, xArray.length, true, xArray);
        double p = xMatrix.mult(theta).get(0, 0);

        return activation.apply(p);
    }

    public Double score(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY) {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        if (dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        int correct = 0;
        for (int i = 0; i < dataX.size(); i++) {
            int prediction = predict(dataX.get(i));
            if (prediction == dataY.get(i).intValue()) {
                correct++;
            }
        }
        return (double) correct / dataX.size();
    }

    @Override
    public String toString() {
        if (theta == null) {
            return "Perceptron (unfitted)";
        }

        StringBuilder sb = new StringBuilder();
        // Header with hyper‑parameters
        sb.append("Perceptron(")
                .append("α=").append(learningRate)
                .append(", iter=").append(iterations)
                .append(", thresh=").append(threshold)
                .append(", shuffle=").append(shuffle)
                .append(", verbose=").append(verbose)
                .append(") ");

        // Weights
        sb.append("weights=[");
        for (int i = 0; i < theta.getNumRows(); i++) {
            sb.append(String.format("%.4f", theta.get(i, 0)));
            if (i < theta.getNumRows() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");

        return sb.toString();
    }
}
