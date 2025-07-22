package com.hindbiswas.ml.models;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

/**
 * Multivariate linear regression using matrix algebra (Ordinary Least Squares)
 */
public class LinearRegressionMultiVar {
    private SimpleMatrix theta = null;

    /**
     * Fits the model to the provided feature matrix and target vector.
     * Accepts any List<List<Double>> and List<Double>.
     * Copies inputs internally to guard against external mutation.
     * 
     * @param dataX List of feature vectors (size m x n)
     * @param dataY List of target values (size m)
     * @return This model instance, for chaining.
     * @throws IllegalArgumentException if sizes differ or inputs are empty.
     */
    public LinearRegressionMultiVar fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException {
        if (dataX == null || dataY == null || dataX.size() != dataY.size() || dataX.isEmpty()) {
            throw new IllegalArgumentException("Data lists must be non-null, of the same non-zero length.");
        }
        int m = dataX.size();
        int features = dataX.get(0).size();
        if (features == 0) {
            throw new IllegalArgumentException("Feature vectors must have at least one feature.");
        }

        // Copy inputs to internal arrays
        // Build X matrix with bias term
        SimpleMatrix x = new SimpleMatrix(m, features + 1);
        SimpleMatrix y = new SimpleMatrix(m, 1);

        for (int i = 0; i < m; i++) {
            ArrayList<Double> row = dataX.get(i);
            if (row.size() != features) {
                throw new IllegalArgumentException("All feature vectors must have the same length.");
            }
            // bias term
            x.set(i, 0, 1.0);
            for (int j = 0; j < features; j++) {
                x.set(i, j + 1, row.get(j));
            }
            y.set(i, 0, dataY.get(i));
        }

        // theta = (X^T * X)^(-1) * X^T * y
        theta = x.transpose().mult(x).invert().mult(x.transpose()).mult(y);
        return this;
    }

    /**
     * Predicts a target value for a given feature vector.
     * 
     * @param xFeatures List of feature values (size n)
     * @return Predicted y
     * @throws IllegalStateException    if fit() has not been called
     * @throws IllegalArgumentException if xFeatures size does not match model
     */
    public Double predict(ArrayList<Double> xFeatures) throws IllegalArgumentException, IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        int features = theta.getNumRows() - 1;
        if (xFeatures == null || xFeatures.size() != features) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", features,
                            (xFeatures == null ? 0 : xFeatures.size())));
        }

        // Build row vector [1, x1, x2, ...]
        double[] arr = new double[features + 1];
        arr[0] = 1.0;
        for (int i = 0; i < features; i++) {
            arr[i + 1] = xFeatures.get(i);
        }
        SimpleMatrix xRow = new SimpleMatrix(1, features + 1, true, arr);
        return xRow.mult(theta).get(0, 0);
    }

    /**
     * Returns the parameter vector theta (size (n+1) x 1).
     * 
     * @return theta
     * @throws IllegalStateException if model unfitted.
     */
    public SimpleMatrix getTheta() throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        return theta;
    }

    @Override
    public String toString() {
        if (theta == null) {
            return "LinearRegressionMat (unfitted model)";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("LinearRegressionMat [y = ");
        // theta is column vector: theta0 + theta1*x1 + ...
        sb.append(String.format("%.4f", theta.get(0, 0)));
        for (int i = 1; i < theta.getNumRows(); i++) {
            sb.append(" + ");
            sb.append(String.format("%.4f", theta.get(i, 0)));
            sb.append("*x").append(i);
        }
        sb.append("]");
        return sb.toString();
    }
}
