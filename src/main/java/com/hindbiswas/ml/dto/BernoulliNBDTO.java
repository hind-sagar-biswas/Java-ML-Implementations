package com.hindbiswas.ml.dto;

/**
 * DTO for Multinomial Naive Bayes model parameters.
 */
public class BernoulliNBDTO extends NaiveBayesDTO {
    /** Log probabilities of features per class. */
    public double[][] featureLogProb;
    /** Log probabilities of features not occurring per class. */
    public double[][] featureLogProbNeg;
}
