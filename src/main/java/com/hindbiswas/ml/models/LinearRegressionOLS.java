package com.hindbiswas.ml.models;

import java.util.ArrayList;

public class LinearRegressionOLS {
    private Double slope;
    private Double intercept;

    private Double meanX;
    private Double meanY;

    private ArrayList<Double> dataX;
    private ArrayList<Double> dataY;

    private void calculateMean() {
        meanX = 0.0;
        meanY = 0.0;
        for (int i = 0; i < dataX.size(); i++) {
            meanX += dataX.get(i);
            meanY += dataY.get(i);
        }
        meanX /= dataX.size();
        meanY /= dataY.size();
    }

    private void calculateCoefficients() {
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < dataX.size(); i++) {
            numerator += (dataX.get(i) - meanX) * (dataY.get(i) - meanY);
            denominator += Math.pow(dataX.get(i) - meanX, 2);
        }
        slope = numerator / denominator;
        intercept = meanY - slope * meanX;
    }

    public LinearRegressionOLS fit(ArrayList<Double> dataX, ArrayList<Double> dataY)
            throws IllegalArgumentException {
        if (dataX.size() != dataY.size() || dataX.size() == 0) {
            throw new IllegalArgumentException("Data arrays must be of the same non-zero length.");
        }

        this.dataX = dataX;
        this.dataY = dataY;

        calculateMean();
        calculateCoefficients();

        return this;
    }

    public Double predict(Double x) {
        if (slope == null || intercept == null) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        return slope * x + intercept;
    }

    public Double getIntercept() {
        return intercept;
    }

    public Double getSlope() {
        return slope;
    }

    public String toString() {
        return "LinearRegression [y = " + slope + "x + " + intercept + "]";
    }
}
