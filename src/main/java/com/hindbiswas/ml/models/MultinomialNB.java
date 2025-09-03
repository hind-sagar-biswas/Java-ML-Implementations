package com.hindbiswas.ml.models;

import java.nio.file.Path;
import java.util.Objects;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.data.DataPoint;
import com.hindbiswas.ml.dto.MultinomialNBDTO;
import com.hindbiswas.ml.util.ModelIO;

/**
 * MultinomialNB
 */
public class MultinomialNB extends NaiveBayes {
    private double[][] featureLogProb;

    public MultinomialNB() {
    }

    public MultinomialNB(MultinomialNBDTO dto) {
        this.alpha = dto.alpha;
        this.classes = dto.classes;
        this.features = dto.features;
        this.logClassPriors = dto.logClassPriors;
        this.featureLogProb = dto.featureLogProb;
        this.fitted = true;
    }

    @Override
    public MultinomialNB fit(DataFrame df) {
        df = Objects.requireNonNull(df, "DataFrame cannot be null");

        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }

        this.setDatasetInfo(df);
        this.calculateFeatureLogProb(df);
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
                logLikelihood += features[f] * featureLogProb[c][f];
            }
            probs[c] = logClassPriors[c] + logLikelihood;
        }

        return this.pickMax(probs);
    }

    @Override
    public MultinomialNBDTO toDTO() {
        MultinomialNBDTO dto = new MultinomialNBDTO();
        dto.alpha = alpha;
        dto.classes = classes;
        dto.features = features;
        dto.logClassPriors = logClassPriors;
        dto.featureLogProb = featureLogProb;
        return dto;
    }

    @Override
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }

    public String toString() {
        MultinomialNBDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }

    public static MultinomialNB importModel(Path path) throws Exception {
        return ModelIO.importModel(path, MultinomialNBDTO.class, MultinomialNB.class);
    }

    private void calculateFeatureLogProb(DataFrame df) {
        int[][] featureCount = new int[this.classes.length][this.features + 1]; // last index is for total sum

        for (DataPoint dp : df) {
            Integer classIndex = classIndices.get(dp.label);
            if (classIndex == null) {
                throw new IllegalArgumentException("Unknown label encountered during stats calc: " + dp.label);
            }
            int ci = classIndex;
            for (int j = 0; j < features; j++) {
                featureCount[ci][j] += (int) dp.features[j];
                featureCount[ci][features] += (int) dp.features[j];
            }
        }

        this.featureLogProb = new double[this.classes.length][this.features];
        for (int i = 0; i < classes.length; i++) {
            for (int j = 0; j < features; j++) {
                this.featureLogProb[i][j] = Math
                        .log((double) (featureCount[i][j] + alpha)
                                / (featureCount[i][features] + this.features * alpha));
            }
        }
    }
}
