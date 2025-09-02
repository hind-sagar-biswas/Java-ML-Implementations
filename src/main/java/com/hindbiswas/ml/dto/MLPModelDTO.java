package com.hindbiswas.ml.dto;

import java.util.ArrayList;

/**
 * MLPModelDTO
 */
public class MLPModelDTO extends DTO {

    public int inputSize;
    public int hiddenLayers;
    public int outputSize;
    public double learningRate;

    public int epochs;
    public int batchSize;
    public double validationSplit;

    public String lossGradientName;
    public String lossFunctionName;

    public ArrayList<LayerDTO> layers = new ArrayList<>();
    public String libraryVersion = "1.0";
    public String schemaVersion = "1";
}
