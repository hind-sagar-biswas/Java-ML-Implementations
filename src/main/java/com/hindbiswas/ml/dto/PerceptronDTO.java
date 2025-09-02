package com.hindbiswas.ml.dto;

/**
 * PerceptronDTO
 */
public class PerceptronDTO extends DTO {
    public double learningRate;
    public int iterations;
    public double threshold;
    public boolean shuffle;
    public boolean verbose;
    public boolean[] theta;
    public int weightSeed;
}
