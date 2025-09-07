package com.hindbiswas.ml.data;

/**
 * Represents a single data point with features and label.
 */
public class DataPoint {
    /** Feature vector for this data point. */
    public final double[] features;
    /** Label for this data point. */
    public final double label;

    /**
     * Constructs a DataPoint with given features and label.
     * @param features Feature vector
     * @param label Label value
     */
    public DataPoint(double[] features, double label) {
        this.features = features;
        this.label = label;
    }
}
