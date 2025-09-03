package com.hindbiswas.ml.models;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.data.DataPoint;
import com.hindbiswas.ml.dto.GaussianNBDTO;
import com.hindbiswas.ml.util.ModelIO;

/**
 * GaussianNB - Gaussian Naive Bayes classifier.
 *
 * <p>
 * Trains per-class, per-feature Gaussian distributions. Means and variances are
 * calculated using Welford's online algorithm for numerical stability.
 * Variances
 * are floored by {@link #VAR_EPS} to avoid divide-by-zero and NaNs.
 * </p>
 *
 * <p>
 * Prediction returns the class (as a {@link Double}) with the highest
 * log-posterior
 * (log prior + sum of log-likelihoods across features). If multiple classes tie
 * within a tiny tolerance, a random class among the tied ones is chosen.
 * </p>
 */
public class GaussianNB extends NaiveBayes {
    private static final double VAR_EPS = 1e-9;
    private double[][] means;
    private double[][] variances;

    public GaussianNB() {
    }

    public GaussianNB(GaussianNBDTO dto) {
        this.alpha = dto.alpha;
        this.classes = dto.classes;
        this.features = dto.features;
        this.logClassPriors = dto.logClassPriors;
        this.means = dto.means;
        this.variances = dto.variances;
        this.fitted = true;
    }

    /**
     * Load an exported model from disk.
     *
     * @param path path to the DTO/JSON/etc produced by
     *             {@link ModelIO#export(Path, Object)}
     * @return deserialized {@link GaussianNB} instance
     * @throws Exception propagated from ModelIO when import fails
     */
    public static GaussianNB importModel(Path path) throws Exception {
        return ModelIO.importModel(path, GaussianNBDTO.class, GaussianNB.class);
    }

    /**
     * Fit the model to the provided dataset. This will compute:
     * <ul>
     * <li>class priors (in the superclass via setDatasetInfo)</li>
     * <li>per-class means and variances (via
     * {@link #calculateStats(DataFrame)})</li>
     * </ul>
     *
     * @param df the training dataset (must be non-null and non-empty)
     * @return this fitted {@link GaussianNB} instance
     * @throws NullPointerException     if {@code df} is null
     * @throws IllegalArgumentException if {@code df} is empty
     */
    @Override
    public NaiveBayes fit(DataFrame df) {
        df = Objects.requireNonNull(df, "DataFrame cannot be null");

        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }

        this.setDatasetInfo(df);
        this.calculateStats(df);
        this.fitted = true;

        return this;
    }

    /**
     * Predict the class for a single feature vector.
     *
     * @param features array of feature values (length must match the training
     *                 feature count)
     * @return predicted class label (as a {@link Double})
     * @throws IllegalStateException    if the model has not been fitted
     * @throws IllegalArgumentException if {@code features} length does not match
     *                                  expected feature count
     */
    @Override
    public Double predict(double[] features) {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted yet.");
        }
        if (features.length != this.features) {
            throw new IllegalArgumentException(
                    String.format("Expected %d features, but got %d.", this.features, features.length));
        }

        double[] probs = new double[classes.length];
        for (int c = 0; c < classes.length; c++) {
            double logLikelihood = 0.0;
            for (int f = 0; f < features.length; f++) {
                double var = Math.max(variances[c][f], VAR_EPS);
                double diff = features[f] - means[c][f];
                logLikelihood += -0.5 * (Math.log(2 * Math.PI * var) + (diff * diff) / var);
            }
            probs[c] = logClassPriors[c] + logLikelihood;
        }

        return this.pickMax(probs);

    }

    /**
     * Convert this model to a serializable DTO.
     *
     * @return DTO containing model parameters suitable for JSON export
     */
    @Override
    public GaussianNBDTO toDTO() {
        GaussianNBDTO dto = new GaussianNBDTO();
        dto.alpha = alpha;
        dto.classes = classes;
        dto.features = features;
        dto.logClassPriors = logClassPriors;
        dto.means = means;
        dto.variances = variances;
        return dto;
    }

    /**
     * Pretty-printed JSON representation (via DTO).
     *
     * @return JSON string
     */
    @Override
    public String toString() {
        GaussianNBDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }

    /**
     * Export this model to disk using {@link ModelIO#export(Path, Object)}.
     *
     * @param path destination path
     * @return true if export succeeded (delegated to ModelIO)
     */
    @Override
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }

    /**
     * Compute per-class means and variances using Welford's online algorithm.
     * If a class has zero samples, its means are set to 0 and variances floored to
     * {@link #VAR_EPS}.
     *
     * @param df dataset to compute statistics from
     * @throws IllegalArgumentException if an unknown label is encountered
     */
    private void calculateStats(DataFrame df) throws IllegalArgumentException {
        this.means = new double[classes.length][features];
        this.variances = new double[classes.length][features];

        double[] count = new double[classes.length];
        double[][] m2 = new double[classes.length][features]; // sum of squares of differences

        // Welford algorithm per class-feature
        for (DataPoint dp : df) {
            Integer classIndex = classIndices.get(dp.label);
            if (classIndex == null) {
                throw new IllegalArgumentException("Unknown label encountered during stats calc: " + dp.label);
            }
            int ci = classIndex;
            count[ci]++;
            double n = count[ci];
            for (int j = 0; j < features; j++) {
                double x = dp.features[j];
                double delta = x - means[ci][j];
                means[ci][j] += delta / n;
                double delta2 = x - means[ci][j];
                m2[ci][j] += delta * delta2;
            }
        }

        for (int c = 0; c < classes.length; c++) {
            if (count[c] > 0) {
                for (int f = 0; f < features; f++) {
                    // variance = M2 / n
                    variances[c][f] = m2[c][f] / count[c];
                    // apply floor for numerical safety
                    if (variances[c][f] < VAR_EPS)
                        variances[c][f] = VAR_EPS;
                }
            } else {
                // Unseen class
                for (int f = 0; f < features; f++) {
                    variances[c][f] = VAR_EPS;
                    means[c][f] = 0.0;
                }
            }
        }
    }
}
