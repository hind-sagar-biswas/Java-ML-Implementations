package com.hindbiswas.ml.models;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class LogisticRegression {
    private SimpleMatrix theta = null;
    private Double learningRate = 0.01;
    private Integer iterations = 1000;

    public LogisticRegression() {
    }

    public LogisticRegression(Double learningRate, Integer iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public LogisticRegression fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException {
        if (dataX.size() != dataY.size() || dataX.size() == 0) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        if (dataX.get(0).size() == 0) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        for (Double value : dataY) {
            if (value != 0 && value != 1)
                throw new IllegalArgumentException("DataY values must be either 0 or 1.");
        }

        int m = dataX.size();
        int n = dataX.get(0).size() + 1;

        SimpleMatrix x = new SimpleMatrix(m, n);
        SimpleMatrix y = new SimpleMatrix(m, 1);

        for (int i = 0; i < m; i++) {
            x.set(i, 0, 1.0); // intercept term
            for (int j = 0; j < n - 1; j++) {
                x.set(i, j + 1, dataX.get(i).get(j));
            }
            y.set(i, 0, dataY.get(i));
        }

        theta = new SimpleMatrix(n, 1); // Initialize theta with zeros

        for (int i = 0; i < iterations; i++) {
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

    public Double predict(ArrayList<Double> x) throws IllegalStateException {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        double[] xArray = new double[x.size() + 1];
        xArray[0] = 1;
        for (int i = 0; i < x.size(); i++) {
            xArray[i + 1] = x.get(i);
        }

        SimpleMatrix xMatrix = new SimpleMatrix(1, xArray.length, true, xArray);
        double z = xMatrix.mult(theta).get(0, 0);
        double p = sigmoid(z);

        return p;
    }

    public Integer classify(ArrayList<Double> x) throws IllegalStateException {
        double p = predict(x);

        return p >= 0.5 ? 1 : 0;
    }

    public SimpleMatrix getTheta() {
        if (theta == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        return theta;
    }

    public String toString() {
        if (theta == null) {
            return "LogisticRegression (unfitted model)";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("LogisticRegression [y = ");
        sb.append(String.format("%.4f", theta.get(0, 0)));
        for (int i = 1; i < theta.getNumRows(); i++) {
            sb.append(" + ").append(String.format("%.4f", theta.get(i, 0))).append("*x").append(i);
        }
        sb.append("]");
        return sb.toString();

    }
}
