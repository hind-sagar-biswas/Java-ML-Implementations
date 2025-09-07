package com.hindbiswas.ml.dto;

/**
 * DTO for Gaussian Naive Bayes model parameters.
 */
public class GaussianNBDTO extends NaiveBayesDTO {
    /** Per-class, per-feature means. */
    public double[][] means;
    /** Per-class, per-feature variances. */
    public double[][] variances;
}
