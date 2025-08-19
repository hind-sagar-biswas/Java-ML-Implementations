package com.hindbiswas.ml.util;

import java.util.ArrayList;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

/**
 * Matrix
 */
public class Matrix {
    private static final Random GLOBAL_RND = new Random();

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

    public static SimpleMatrix row(double[] data) {
        double[] dataArray = new double[data.length + 1];
        dataArray[0] = 1.0;
        for (int i = 0; i < data.length; i++) {
            dataArray[i + 1] = data[i];
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

    public static SimpleMatrix column(double[] data) {
        double[] dataArray = new double[data.length + 1];
        dataArray[0] = 1.0;
        for (int i = 0; i < data.length; i++) {
            dataArray[i + 1] = data[i];
        }
        return new SimpleMatrix(dataArray.length, 1, true, dataArray);
    }

    public static SimpleMatrix columnWithoutBias(ArrayList<Double> data) {
        double[] dataArray = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            dataArray[i] = data.get(i);
        }
        return new SimpleMatrix(dataArray.length, 1, true, dataArray);
    }

    public static SimpleMatrix columnWithoutBias(double[] dataArray) {
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

    public static SimpleMatrix xavier(int m, int n) {
        SimpleMatrix matrix = new SimpleMatrix(m, n);
        double scale = Math.sqrt(2.0 / (m + n - 1));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                matrix.set(i, j, GLOBAL_RND.nextGaussian() * scale);
            }
        }
        return matrix;
    }

    public static double[][] toArray2D(SimpleMatrix m) {
        int r = m.getNumRows(), c = m.getNumCols();
        double[][] a = new double[r][c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                a[i][j] = m.get(i, j);
        return a;
    }

    public static SimpleMatrix fromArray2D(double[][] a) {
        return new SimpleMatrix(a);
    }
}
