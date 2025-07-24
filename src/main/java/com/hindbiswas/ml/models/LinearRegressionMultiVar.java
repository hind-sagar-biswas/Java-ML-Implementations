package com.hindbiswas.ml.models;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import com.hindbiswas.ml.util.Matrix;

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
        SimpleMatrix x = Matrix.build(dataX);
        SimpleMatrix y = new SimpleMatrix(m, 1);

        for (int i = 0; i < m; i++) {
            y.set(i, 0, dataY.get(i));
        }

        // theta = (X^T * X)^(-1) * X^T * y
        theta = x.transpose().mult(x).invert().mult(x.transpose()).mult(y);
        return this;
    }

    /**
     * Predicts a target value for a given feature vector.
     * 
     * @param x List of feature values (size n)
     * @return Predicted y
     * @throws IllegalStateException    if fit() has not been called
     * @throws IllegalArgumentException if xFeatures size does not match model
     */
    public Double predict(ArrayList<Double> x) throws IllegalArgumentException, IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        int features = theta.getNumRows() - 1;
        if (x == null || x.size() != features) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", features,
                            (x == null ? 0 : x.size())));
        }

        SimpleMatrix xRow = Matrix.row(x);
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
