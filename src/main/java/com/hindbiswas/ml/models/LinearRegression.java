package com.hindbiswas.ml.models;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

public class LinearRegression {
    private SimpleMatrix coefficients = null;

    public LinearRegression fit(ArrayList<ArrayList<Double>> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException {
        if (dataX.size() != dataY.size() || dataX.size() == 0) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        if (dataX.get(0).size() == 0) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        SimpleMatrix x = new SimpleMatrix(dataX.size(), dataX.get(0).size() + 1);
        SimpleMatrix y = new SimpleMatrix(dataY.size(), 1);

        for (int i = 0; i < dataX.size(); i++) {
            for (int j = 0; j < dataX.get(0).size(); j++) {
                x.set(i, j + 1, dataX.get(i).get(j));
            }
            x.set(i, 0, 1);
            y.set(i, 0, dataY.get(i));
        }

        // coefficients = (X^T * X)^(-1) * X^T * y
        coefficients = x.transpose().mult(x).invert().mult(x.transpose()).mult(y);
        return this;
    }

    public Double predict(ArrayList<Double> x) {
        if (coefficients == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        double[] xArray = new double[x.size() + 1];
        xArray[0] = 1;
        for (int i = 0; i < x.size(); i++) {
            xArray[i + 1] = x.get(i);
        }

        SimpleMatrix xMatrix = new SimpleMatrix(1, x.size() + 1, true, xArray);
        return xMatrix.mult(coefficients).get(0, 0);
    }

    public SimpleMatrix getCoefficients() {
        if (coefficients == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        return coefficients;
    }

    public String toString() {
        return "LinearRegression [y = " + coefficients + "]";
    }
}
