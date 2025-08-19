package com.hindbiswas.ml.data;

public class DataPoint {
    public final double[] features;
    public final double label;

    public DataPoint(double[] features, double label) {
        this.features = features;
        this.label = label;
    }
}
