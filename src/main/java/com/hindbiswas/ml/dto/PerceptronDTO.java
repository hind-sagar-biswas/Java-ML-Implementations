package com.hindbiswas.ml.dto;

/**
 * DTO for Perceptron model parameters and configuration.
 */
public class PerceptronDTO extends DTO {
    /** Learning rate for weight updates. */
    public double learningRate;
    /** Number of training iterations. */
    public int iterations;
    /** Activation threshold. */
    public double threshold;
    /** Whether to shuffle data each epoch. */
    public boolean shuffle;
    /** Enable verbose logging. */
    public boolean verbose;
    /** Weight vector (theta). */
    public boolean[] theta;
    /** Random seed for weight initialization. */
    public int weightSeed;
}
