package com.hindbiswas.ml.models;

import java.util.Map;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Objects;

import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.data.DataPoint;
import com.hindbiswas.ml.util.ModelIO;

/**
 * NaiveBayes
 */
abstract public class NaiveBayes implements Model {
    protected double alpha = 1;
    protected boolean fitted = false;
    protected int features;
    protected double[] classes;
    protected Map<Double, Integer> classIndices = new HashMap<>();
    protected double[] logClassPriors;

    /**
     * Set the alpha parameter.
     *
     * @param alpha alpha parameter
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Fit the model to the given {@link DataFrame}.
     *
     * @param df training dataframe
     * @return the fitted model
     * @throws IllegalArgumentException if the dataframe is null, empty, or has
     *                                  inconsistent feature count
     */
    public abstract NaiveBayes fit(DataFrame df) throws IllegalArgumentException;

    /**
     * Predict the class probabilities for the given features.
     *
     * @param features features to predict
     * @return predicted class probabilities
     * @throws IllegalArgumentException if features size does not match model
     * @throws IllegalStateException    if the model has not been fitted
     */
    public abstract Double predict(double[] features) throws IllegalArgumentException, IllegalStateException;

    /**
     * Compute classification accuracy of the model on a given {@link DataFrame}.
     *
     * <p>
     * Accuracy is computed by taking the argmax of the raw outputs for each row and
     * comparing it to the integer label in the dataframe.
     * </p>
     *
     * @param df evaluation dataframe
     * @return accuracy in [0.0, 1.0]
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if the dataframe is empty, or has
     *                                  wrong feature count
     * @throws NullPointerException     if the dataframe is null
     */
    @Override
    public double score(DataFrame df)
            throws IllegalArgumentException, IllegalStateException, NullPointerException {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }

        df = Objects.requireNonNull(df, "DataFrame is null.");
        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }
        if (df.featureCount() != features) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but DataFrame has %d.", features, df.featureCount()));
        }

        int correct = 0;
        for (DataPoint dp : df) {
            int actual = (int) dp.label;
            Double predicted = predict(dp.features);

            if (classIndices.get(predicted) == actual) {
                correct++;
            }
        }
        return (double) correct / df.size();
    }

    /**
     * Set the class mapping from the given labels.
     * 
     * @param labels training labels
     */
    protected void setDatasetInfo(DataFrame df) {
        features = df.featureCount();
        classes = df.getUniqueLabels();
        for (int i = 0; i < classes.length; i++) {
            classIndices.put(classes[i], i);
        }
        logClassPriors = setPriors(df.getLabels());
    }

    /**
     * Sets the class priors from the given labels.
     *
     * @param labels training labels
     * @return class priors
     * @throws IllegalStateException if classes are not set
     */
    private double[] setPriors(double[] labels) throws IllegalStateException {
        if (classes == null) {
            throw new IllegalStateException("Classes not set.");
        }

        double[] out = new double[classes.length];
        int[] classCounts = new int[classes.length];

        for (int i = 0; i < labels.length; i++) {
            Integer idx = classIndices.get(labels[i]);
            if (idx == null) {
                throw new IllegalStateException("Unknown label encountered when setting priors: " + labels[i]);
            }
            classCounts[idx]++;
        }

        double alpha = 1e-9;
        double denom = labels.length + alpha * classes.length;
        for (int i = 0; i < classes.length; i++) {
            out[i] = Math.log((classCounts[i] + alpha) / denom);
        }

        return out;
    }

    /**
     * Export the model (DTO JSON) to the given file path.
     *
     * @param path output path
     * @return true on success, false on failure
     */
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }
}
