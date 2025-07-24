package com.hindbiswas.ml.models;

import java.util.ArrayList;
import java.util.Objects;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

/**
 * Implements multivariate linear regression using batch gradient descent.
 *
 * This class supports fitting a model to data via gradient descent,
 * making predictions, and retrieving the learned parameters.
 */
public class LinearRegressionGD {
    private Double learningRate = 0.01;
    private Integer iterations = 1000;
    private SimpleMatrix theta = null;

    /**
     * Creates a LinearRegressionGD with default learning rate (0.01) and iterations
     * (1000).
     */
    public LinearRegressionGD() {
    }

    /**
     * Creates a LinearRegressionGD with specified hyperparameters.
     *
     * @param learningRate step size for gradient updates (must be > 0)
     * @param iterations   number of gradient descent iterations (must be > 0)
     * @throws IllegalArgumentException if learningRate &le; 0 or iterations &le; 0
     */
    public LinearRegressionGD(Double learningRate, Integer iterations) throws IllegalArgumentException {
        if (learningRate == null || learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be > 0");
        }
        if (iterations == null || iterations <= 0) {
            throw new IllegalArgumentException("iterations must be > 0");
        }
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    /**
     * Fits the linear regression model to the provided training data using batch
     * gradient descent.
     *
     * @param dataX list of feature vectors (size m x n)
     * @param dataY list of target values (size m)
     * @return this model instance (for chaining)
     * @throws IllegalArgumentException if inputs are null, lengths mismatch, or no
     *                                  features
     */
    public LinearRegressionGD fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException {
        Objects.requireNonNull(dataX, "dataX cannot be null");
        Objects.requireNonNull(dataY, "dataY cannot be null");
        if (dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }
        if (dataX.get(0).isEmpty()) {
            throw new IllegalArgumentException("Feature vectors must contain at least one feature.");
        }

        int m = dataX.size();
        int n = dataX.get(0).size() + 1; // +1 for intercept

        // Build design matrix X and target vector y
        SimpleMatrix x = Matrix.build(dataX);
        SimpleMatrix y = new SimpleMatrix(m, 1);
        for (int i = 0; i < m; i++) {
            y.set(i, 0, dataY.get(i));
        }

        // Initialize theta to zeros
        theta = new SimpleMatrix(n, 1);

        // Perform gradient descent
        for (int iter = 0; iter < iterations; iter++) {
            SimpleMatrix predictions = x.mult(theta);
            SimpleMatrix errors = predictions.minus(y);
            SimpleMatrix gradient = x.transpose().mult(errors).divide((double) m);
            theta = theta.minus(gradient.scale(learningRate));
        }

        return this;
    }

    /**
     * Predicts the target value for a single feature vector.
     *
     * @param features list of feature values (size n)
     * @return predicted target value
     * @throws IllegalStateException    if fit() has not been called
     * @throws IllegalArgumentException if features is null or length mismatch
     */
    public Double predict(ArrayList<Double> x) throws IllegalArgumentException, IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        Objects.requireNonNull(x, "features cannot be null");
        if (x.size() + 1 != theta.getNumRows()) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", theta.getNumRows() - 1, x.size()));
        }

        SimpleMatrix xMatrix = Matrix.row(x);
        return xMatrix.mult(theta).get(0, 0);
    }

    /**
     * Returns the learned parameter vector (theta).
     *
     * @return parameter vector of size (n+1) x 1
     * @throws IllegalStateException if the model is unfitted
     */
    public SimpleMatrix getTheta() throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        return theta;
    }

    /**
     * Returns a string representation of the regression equation.
     *
     * @return human-readable equation: y = theta0 + theta1*x1 + ...
     */
    @Override
    public String toString() {
        if (theta == null) {
            return "LinearRegressionGD (unfitted model)";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("LinearRegressionGD [y = ");
        sb.append(String.format("%.4f", theta.get(0, 0)));
        for (int i = 1; i < theta.getNumRows(); i++) {
            sb.append(" + ").append(String.format("%.4f", theta.get(i, 0))).append("*x").append(i);
        }
        sb.append("]");
        return sb.toString();
    }
}
