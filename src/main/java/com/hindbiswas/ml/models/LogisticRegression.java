package com.hindbiswas.ml.models;

import java.util.ArrayList;
import org.ejml.simple.SimpleMatrix;

/**
 * Implements logistic regression for binary classification using batch gradient
 * descent.
 * <p>
 * Supports configurable learning rate and number of iterations.
 * Predictions return probabilities via sigmoid, and classifications threshold
 * at 0.5.
 */
public class LogisticRegression {
    /** Weight vector (theta), including intercept term, set after fitting. */
    private SimpleMatrix theta = null;
    /** Learning rate (α) for gradient descent updates. */
    private Double learningRate = 0.01;
    /** Number of iterations (epochs) for training. */
    private Integer iterations = 1000;

    /**
     * Default constructor using learningRate=0.01 and iterations=1000.
     */
    public LogisticRegression() {
    }

    /**
     * Constructor with user-specified learning rate and iterations.
     *
     * @param learningRate step size for gradient updates
     * @param iterations   number of epochs to run
     */
    public LogisticRegression(Double learningRate, Integer iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    /**
     * Computes the sigmoid activation function.
     *
     * @param z raw input value
     * @return sigmoid(z) in range (0,1)
     */
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Trains the logistic regression model on the provided dataset.
     *
     * @param dataX list of feature vectors (each inner list is one example)
     * @param dataY list of binary labels (0 or 1) matching dataX size
     * @return this instance with trained parameters
     * @throws IllegalArgumentException if data sizes mismatch, are empty, or labels
     *                                  invalid
     */
    public LogisticRegression fit(
            ArrayList<ArrayList<Double>> dataX,
            ArrayList<Double> dataY) throws IllegalArgumentException {
        if (dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }
        if (dataX.get(0).isEmpty()) {
            throw new IllegalArgumentException("Feature vectors must be non-empty.");
        }
        for (Double value : dataY) {
            if (value != 0 && value != 1) {
                throw new IllegalArgumentException("DataY values must be either 0 or 1.");
            }
        }

        int m = dataX.size();
        int n = dataX.get(0).size() + 1; // +1 for intercept term

        // Construct design matrix X and label vector y
        SimpleMatrix x = new SimpleMatrix(m, n);
        SimpleMatrix y = new SimpleMatrix(m, 1);
        for (int i = 0; i < m; i++) {
            x.set(i, 0, 1.0); // intercept term
            for (int j = 0; j < n - 1; j++) {
                x.set(i, j + 1, dataX.get(i).get(j));
            }
            y.set(i, 0, dataY.get(i));
        }

        // Initialize parameters to zero
        theta = new SimpleMatrix(n, 1);

        // Batch gradient descent
        for (int iter = 0; iter < iterations; iter++) {
            SimpleMatrix z = x.mult(theta);
            SimpleMatrix predictions = new SimpleMatrix(m, 1);
            for (int j = 0; j < m; j++) {
                predictions.set(j, 0, sigmoid(z.get(j, 0)));
            }
            SimpleMatrix error = predictions.minus(y);
            SimpleMatrix gradient = x.transpose().mult(error).divide(m);
            theta = theta.minus(gradient.scale(learningRate));
        }

        return this;
    }

    /**
     * Computes the probability estimate for a single feature vector.
     *
     * @param x feature vector (excluding intercept term)
     * @return probability in [0,1]
     * @throws IllegalStateException if model has not been fitted
     */
    public Double predict(ArrayList<Double> x) throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        double[] xArray = new double[x.size() + 1];
        xArray[0] = 1.0;
        for (int i = 0; i < x.size(); i++) {
            xArray[i + 1] = x.get(i);
        }
        SimpleMatrix xMatrix = new SimpleMatrix(1, xArray.length, true, xArray);
        double z = xMatrix.mult(theta).get(0, 0);
        return sigmoid(z);
    }

    /**
     * Classifies a single feature vector using 0.5 probability threshold.
     *
     * @param x feature vector (excluding intercept term)
     * @return predicted class label (0 or 1)
     * @throws IllegalStateException if model has not been fitted
     */
    public Integer classify(ArrayList<Double> x) throws IllegalStateException {
        double p = predict(x);
        return p >= 0.5 ? 1 : 0;
    }

    /**
     * Returns the fitted model parameters theta.
     *
     * @return parameter vector including intercept term
     * @throws IllegalStateException if model has not been fitted
     */
    public SimpleMatrix getTheta() throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        return theta;
    }

    /**
     * Generates a human-readable representation of the model and parameters.
     *
     * @return string "LogisticRegression [...]" or unfitted indicator
     */
    @Override
    public String toString() {
        if (theta == null) {
            return "LogisticRegression (unfitted model)";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("LogisticRegression [y = ");
        sb.append(String.format("%.4f", theta.get(0, 0)));
        for (int i = 1; i < theta.getNumRows(); i++) {
            sb.append(" + ")
                    .append(String.format("%.4f", theta.get(i, 0)))
                    .append("*x").append(i);
        }
        sb.append("]");
        return sb.toString();
    }
}
