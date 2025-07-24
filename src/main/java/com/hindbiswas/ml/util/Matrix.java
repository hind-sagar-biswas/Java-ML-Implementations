package com.hindbiswas.ml.util;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

/**
 * Matrix
 */
public class Matrix {
    public static SimpleMatrix build(ArrayList<ArrayList<Double>> data) throws IllegalArgumentException {
        int rows = data.size();
        int cols = data.get(0).size() + 1;
        SimpleMatrix matrix = new SimpleMatrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            if (data.get(i).size() != cols - 1) {
                throw new IllegalArgumentException("All rows must have the same number of columns.");
            }

            matrix.set(i, 0, 1);
            for (int j = 0; j < data.get(0).size(); j++) {
                matrix.set(i, j + 1, data.get(i).get(j));
            }
        }
        return matrix;
    }

    public static SimpleMatrix row(ArrayList<Double> data) {
        double[] dataArray = new double[data.size() + 1];
        dataArray[0] = 1.0;
        for (int i = 0; i < data.size(); i++) {
            dataArray[i + 1] = data.get(i);
        }
        return new SimpleMatrix(1, dataArray.length, true, dataArray);
    }

    public static SimpleMatrix column(ArrayList<Double> data) {
        double[] dataArray = new double[data.size() + 1];
        dataArray[0] = 1.0;
        for (int i = 0; i < data.size(); i++) {
            dataArray[i + 1] = data.get(i);
        }
        return new SimpleMatrix(dataArray.length, 1, true, dataArray);
    }

    public static SimpleMatrix random(int m, int n, long seed) {
        Random rand = new Random(seed);
        SimpleMatrix matrix = new SimpleMatrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                matrix.set(i, j, rand.nextDouble());
            }
        }
        return matrix;
    }
}
