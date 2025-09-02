package com.hindbiswas.ml.dto;

/**
 * NaiveBayesDTO
 */
abstract class NaiveBayesDTO extends DTO {
    public double alpha;
    public int features;
    public double[] classes;
    public double[] logClassPriors;
}
