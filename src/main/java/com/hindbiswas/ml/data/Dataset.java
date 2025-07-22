package com.hindbiswas.ml.data;

import java.util.ArrayList;

public class Dataset {
    public final ArrayList<ArrayList<Double>> X;
    public final ArrayList<Double> y;

    public Dataset(ArrayList<ArrayList<Double>> X, ArrayList<Double> y) {
        this.X = X;
        this.y = y;
    }
}
