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
 * GaussianNB
 */
public class GaussianNB extends NaiveBayes {
    private static final double VAR_EPS = 1e-9;

    public static GaussianNB importModel(Path path) throws Exception {
        return ModelIO.importModel(path, GaussianNBDTO.class, GaussianNB.class);
    }

    private double[][] means;
    private double[][] variances;

    private final Random rng = new Random();

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

        // Find max & ties
        double max = Double.NEGATIVE_INFINITY;
        ArrayList<Integer> maxIndices = new ArrayList<>();
        double tol = 1e-12;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > max + tol) {
                max = probs[i];
                maxIndices.clear();
                maxIndices.add(i);
            } else if (Math.abs(probs[i] - max) <= tol) {
                maxIndices.add(i);
            }
        }

        int pickIndex = maxIndices.get(rng.nextInt(maxIndices.size()));
        return classes[pickIndex];
    }

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

    @Override
    public String toString() {
        GaussianNBDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }

    @Override
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }

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
