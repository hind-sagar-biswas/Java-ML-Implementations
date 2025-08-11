/**
 * Perceptron classifier for binary labels {-1, +1} using the perceptron learning rule.
 * <p>
 * Supports configurable learning rate, number of iterations, activation threshold, 
 * data shuffling, weight initialization seed, and verbose logging.
 */
package com.hindbiswas.ml.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

/**
 * Implements the classic perceptron algorithm for linearly separable data.
 */
public class Perceptron {
    /** Learning rate (α) used to scale weight updates. */
    private Double learningRate = 0.01;
    /** Maximum number of epochs to run if data never perfectly separates. */
    private Integer iterations = 1000;
    /** Threshold for activation: raw >= threshold maps to +1, otherwise -1. */
    private Double threshold = 0d;
    /** Whether to shuffle training data each epoch. */
    private Boolean shuffle = false;
    /** Whether to print weight vector after each epoch. */
    private Boolean verbose = false;
    /** Weight vector (theta), including bias as the first element. */
    private SimpleMatrix theta = null;
    /** Optional seed for random weight initialization. */
    private Integer weightSeed = null;

    /** Activation function mapping a raw dot-product to {-1, +1}. */
    private Activation activation = x -> x >= threshold ? 1 : -1;

    /**
     * Default constructor with default hyperparameters.
     */
    public Perceptron() {
    }

    /**
     * Full constructor with all hyperparameters.
     * 
     * @param learningRate step size for weight updates
     * @param iterations   maximum epochs
     * @param threshold    activation threshold
     * @param shuffle      whether to shuffle each epoch
     * @param verbose      whether to log weights per epoch
     */
    public Perceptron(Double learningRate, Integer iterations, Double threshold, Boolean shuffle, Boolean verbose) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.threshold = threshold;
        this.shuffle = shuffle;
        this.verbose = verbose;
    }

    /**
     * Constructor specifying learning rate, iterations, and threshold.
     * 
     * @param learningRate step size for weight updates
     * @param iterations   maximum epochs
     * @param threshold    activation threshold
     */
    public Perceptron(Double learningRate, Integer iterations, Double threshold) {
        this(learningRate, iterations, threshold, false, false);
    }

    /**
     * Constructor specifying learning rate and threshold.
     * 
     * @param learningRate step size for weight updates
     * @param threshold    activation threshold
     */
    public Perceptron(Double learningRate, Double threshold) {
        this(learningRate, null, threshold, false, false);
    }

    /**
     * Constructor specifying only threshold.
     * 
     * @param threshold activation threshold
     */
    public Perceptron(Double threshold) {
        this(null, null, threshold, false, false);
    }

    /**
     * Sets a seed for random weight initialization. Must be called before fit().
     * 
     * @param seed integer seed
     * @return this perceptron instance for chaining
     * @throws IllegalStateException if weights have already been initialized
     */
    public Perceptron randomizeWeights(int seed) {
        if (theta != null) {
            throw new IllegalStateException("Model has already been fitted. Cannot randomize weights.");
        }
        this.weightSeed = seed;
        return this;
    }

    /**
     * Enables or disables verbose logging of the weight vector after each epoch.
     * 
     * @param verbose true to print weights each epoch
     * @return this perceptron instance for chaining
     */
    public Perceptron verbose(Boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * Enables or disables data shuffling each epoch. Must be called before fit().
     * 
     * @param shuffle true to shuffle training examples each epoch
     * @return this perceptron instance for chaining
     * @throws IllegalStateException if model has been fitted already
     */
    public Perceptron shuffle(Boolean shuffle) {
        if (theta != null) {
            throw new IllegalStateException("Model has already been fitted. Cannot shuffle data.");
        }
        this.shuffle = shuffle;
        return this;
    }

    /**
     * Overrides the default activation function.
     * 
     * @param act custom activation function
     * @return this perceptron instance for chaining
     */
    public Perceptron withActivation(Activation act) {
        this.activation = act;
        return this;
    }

    /**
     * Trains the perceptron on provided features and labels.
     * 
     * @param dataX list of feature vectors (each inner list is one example)
     * @param dataY list of labels, each must be -1 or +1
     * @return this perceptron instance
     * @throws IllegalArgumentException if data sizes mismatch, empty, or invalid
     *                                  labels
     */
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
                SimpleMatrix inputs = Matrix.column(dataX.get(idx));
                double expected = dataY.get(idx);

                // raw activation
                double raw = theta.transpose().mult(inputs).get(0, 0);
                int prediction = (int) activation.apply(raw);
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

    /**
     * Initializes the weight vector (theta) to zeros or random values if a seed is
     * set.
     * 
     * @param n dimension of theta (including bias)
     */
    private void seedWeights(int n) {
        theta = (weightSeed == null) ? new SimpleMatrix(n, 1) : Matrix.random(n, 1, weightSeed);
    }

    /**
     * Predicts the label for a single feature vector.
     * 
     * @param x feature vector (without bias term)
     * @return predicted label (+1 or -1)
     * @throws IllegalStateException if model has not been fitted
     */
    public Integer predict(ArrayList<Double> x) throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        SimpleMatrix xMatrix = Matrix.row(x);
        double p = xMatrix.mult(theta).get(0, 0);

        return (int) activation.apply(p);
    }

    /**
     * Computes classification accuracy on a labeled dataset.
     * 
     * @param dataX list of feature vectors
     * @param dataY list of true labels
     * @return fraction of correct predictions in [0.0, 1.0]
     * @throws IllegalStateException if model has not been fitted
     */
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

    /**
     * Returns a human-readable summary of the perceptron, including hyperparameters
     * and weights.
     * 
     * @return formatted string representation
     */
    @Override
    public String toString() {
        if (theta == null) {
            return "Perceptron (unfitted)";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Perceptron(")
                .append("α=").append(learningRate)
                .append(", iter=").append(iterations)
                .append(", thresh=").append(threshold)
                .append(", shuffle=").append(shuffle)
                .append(", verbose=").append(verbose)
                .append(") weights=[");

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
