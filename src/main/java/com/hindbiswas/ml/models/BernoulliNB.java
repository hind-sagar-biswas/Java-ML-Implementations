package com.hindbiswas.ml.models;

import java.nio.file.Path;
import java.util.Objects;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.data.DataPoint;
import com.hindbiswas.ml.dto.BernoulliNBDTO;
import com.hindbiswas.ml.util.ModelIO;

/**
 * Implementation of the Bernoulli Naive Bayes classifier.
 * 
 * <p>
 * This model is typically used for discrete data such as text classification
 * (bag-of-words). It estimates class conditional probabilities of features
 * based on presence and absence and applies Laplace smoothing.
 * </p>
 * 
 * 
 */
public class BernoulliNB extends NaiveBayes {
    private double[][] featureLogProb;
    private double[][] featureLogProbNeg;

    public BernoulliNB() {
    }

    /**
     * Constructor to restore model state from a Data Transfer Object.
     *
     * @param dto The {@link BernoulliNBDTO} containing model parameters.
     */
    public BernoulliNB(BernoulliNBDTO dto) {
        this.alpha = dto.alpha;
        this.classes = dto.classes;
        this.features = dto.features;
        this.logClassPriors = dto.logClassPriors;
        this.featureLogProb = dto.featureLogProb;
        this.featureLogProbNeg = dto.featureLogProbNeg;
        this.fitted = true;
    }

    /**
     * Fits the Bernoulli Naive Bayes model to the given dataset.
     *
     * @param df The {@link DataFrame} containing labeled training data.
     * @return This model instance after fitting.
     * @throws IllegalArgumentException If the DataFrame is empty.
     * @throws NullPointerException     If the DataFrame is null.
     */
    @Override
    public BernoulliNB fit(DataFrame df) {
        df = Objects.requireNonNull(df, "DataFrame cannot be null");

        if (df.size() == 0) {
            throw new IllegalArgumentException("DataFrame is empty.");
        }

        this.setDatasetInfo(df);
        this.calculateFeatureLogProb(df);
        this.fitted = true;

        return this;
    }

    /**
     * Predicts the most likely class for a given feature vector.
     *
     * @param features Feature array of the same length as training features.
     * @return The predicted class label index.
     * @throws IllegalStateException    If the model has not been fitted yet.
     * @throws IllegalArgumentException If the input feature length is invalid.
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
                if (features[f] == 1.0) {
                    logLikelihood += featureLogProb[c][f];
                } else {
                    logLikelihood += featureLogProbNeg[c][f];
                }
            }
            probs[c] = logClassPriors[c] + logLikelihood;
        }

        return this.pickMax(probs);
    }

    /**
     * Converts the model into a serializable Data Transfer Object.
     *
     * @return A {@link BernoulliNBDTO} containing model parameters.
     */
    @Override
    public BernoulliNBDTO toDTO() {
        BernoulliNBDTO dto = new BernoulliNBDTO();
        dto.alpha = alpha;
        dto.classes = classes;
        dto.features = features;
        dto.logClassPriors = logClassPriors;
        dto.featureLogProb = featureLogProb;
        dto.featureLogProbNeg = featureLogProbNeg;
        return dto;
    }

    /**
     * Exports the model to a file in JSON format.
     *
     * @param path The path to save the model file.
     * @return {@code true} if export succeeds, {@code false} otherwise.
     */
    @Override
    public boolean export(Path path) {
        return ModelIO.export(path, this);
    }

    /**
     * Returns a pretty-printed JSON representation of the model.
     *
     * @return JSON string representation of the model.
     */
    @Override
    public String toString() {
        BernoulliNBDTO dto = toDTO();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(dto);
    }

    /**
     * Imports a previously exported BernoulliNB model from disk.
     *
     * @param path Path to the model file.
     * @return A restored {@link BernoulliNB} instance.
     * @throws Exception If model loading fails.
     */
    public static BernoulliNB importModel(Path path) throws Exception {
        return ModelIO.importModel(path, BernoulliNBDTO.class, BernoulliNB.class);
    }

    /**
     * Calculates the log-probabilities of features given classes using Laplace
     * smoothing.
     *
     * @param df The training {@link DataFrame}.
     * @throws IllegalArgumentException If an unknown class label is encountered.
     */
    private void calculateFeatureLogProb(DataFrame df) {
        int[][] featureCount = new int[this.classes.length][this.features + 1]; // last index is for total sum

        for (DataPoint dp : df) {
            Integer classIndex = classIndices.get(dp.label);
            if (classIndex == null) {
                throw new IllegalArgumentException("Unknown label encountered during stats calc: " + dp.label);
            }
            int ci = classIndex;
            featureCount[ci][features]++;
            for (int j = 0; j < features; j++) {
                if (dp.features[j] == 0)
                    continue;
                featureCount[ci][j]++;
            }
        }

        this.featureLogProb = new double[this.classes.length][this.features];
        this.featureLogProbNeg = new double[this.classes.length][this.features];
        for (int i = 0; i < classes.length; i++) {
            for (int j = 0; j < features; j++) {
                double p = (featureCount[i][j] + alpha) / (featureCount[i][features] + 2 * alpha);
                featureLogProb[i][j] = Math.log(p);
                featureLogProbNeg[i][j] = Math.log(1 - p);
            }
        }
    }
}
