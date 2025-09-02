package com.hindbiswas.ml.models;

import java.nio.file.Path;

import com.hindbiswas.ml.data.DataFrame;
import com.hindbiswas.ml.dto.DTO;

/**
 * Model
 */
public interface Model {

    public Model fit(DataFrame data);

    public double score(DataFrame data);

    public Object predict(double[] features);

    public boolean export(Path path);

    public DTO toDTO();
}
