package com.hindbiswas.ml.dto;

/**
 * DTO for Multinomial Naive Bayes model parameters.
 */
public class MultinomialNBDTO extends NaiveBayesDTO {
    /** Log probabilities of features per class. */
    public double[][] featureLogProb;
}
