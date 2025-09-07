package com.hindbiswas.ml.dto;

/**
 * Base DTO for Naive Bayes models, containing common parameters.
 */
abstract class NaiveBayesDTO extends DTO {
    /** Laplace smoothing parameter. */
    public double alpha;
    /** Number of features. */
    public int features;
    /** Array of class labels. */
    public double[] classes;
    /** Log prior probabilities for each class. */
    public double[] logClassPriors;
}
