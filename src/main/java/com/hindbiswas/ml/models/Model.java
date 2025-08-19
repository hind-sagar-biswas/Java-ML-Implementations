package com.hindbiswas.ml.models;

import com.hindbiswas.ml.data.DataFrame;

/**
 * Model
 */
public interface Model {

    public Model fit(DataFrame data);

    public double score(DataFrame data);

    public Object predict(double[] features);
}
