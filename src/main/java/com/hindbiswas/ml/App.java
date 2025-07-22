package com.hindbiswas.ml;

import java.util.ArrayList;
import java.util.Collections;

import com.hindbiswas.ml.models.LinearRegression;
import com.hindbiswas.ml.models.LinearRegressionMultiVar;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) {
        LinearRegressionMultiVar model = LinearRegression.multi();
        ArrayList<ArrayList<Double>> dataX = new ArrayList<>();
        ArrayList<Double> dataY = new ArrayList<>();

        double[][] xs = {
                { 5, 7, 10 },
                { 7, 8, 9 },
                { 8, 7, 6 },
                { 7, 2, 9 },
                { 2, 17, 4 },
                { 17, 2, 11 },
                { 2, 9, 12 },
                { 9, 6, 9 },
                { 6, 4, 4 },
                { 4, 11, 78 },
                { 11, 12, 77 },
                { 12, 9, 85 },
                { 9, 6, 86 }
        };
        double[] ys = { 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 };
        for (int i = 0; i < xs.length; i++) {
            ArrayList<Double> x = new ArrayList<>();
            for (int j = 0; j < xs[i].length; j++) {
                x.add(xs[i][j]);
            }
            dataX.add(x);
            dataY.add(ys[i]);
        }

        model.fit(dataX, dataY);

        System.out.println(model);
    }
}
