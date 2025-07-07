package com.hindbiswas.ml;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

import com.hindbiswas.ml.models.LinearRegression;

/**
 * Unit test for simple App.
 */
public class LinearRegressionTest {
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldPredictCorrectly() {
        // Prepare training data: y = 2*x + 3
        ArrayList<ArrayList<Double>> dataX = new ArrayList<>();
        ArrayList<Double> dataY = new ArrayList<>();

        // sample points: x = 1,2,3,4,5
        for (double xv : Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)) {
            dataX.add(new ArrayList<>(Arrays.asList(xv)));
            dataY.add(2 * xv + 3);
        }

        // Instantiate and fit the model
        LinearRegression model = new LinearRegression();
        model.fit(dataX, dataY);

        // Test prediction for x = 6
        ArrayList<Double> newX = new ArrayList<>(Arrays.asList(6.0));
        double pred = model.predict(newX);

        // Expected ≈ 15.0
        assertTrue(Math.round(pred) == 15.0);
    }

    @Test
    public void shouldGetCorrectCoefficients() {
        ArrayList<ArrayList<Double>> dataX = new ArrayList<>();
        ArrayList<Double> dataY = new ArrayList<Double>();

        double[][] data = {
                { 5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6 },
                { 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 }
        };

        for (int i = 0; i < data[0].length; i++) {
            dataX.add(new ArrayList<>(Arrays.asList(data[0][i])));
            dataY.add(data[1][i]);
        }

        LinearRegression model = new LinearRegression();
        model.fit(dataX, dataY);

        SimpleMatrix coefficients = model.getCoefficients();
        double b = coefficients.get(0, 0);
        double m = coefficients.get(1, 0);

        // LinearRegression [y = -1.7512877115526122x + 103.10596026490066]
        assertTrue(Math.round(m * 100.0) / 100.0 == -1.75);
        assertTrue(Math.round(b * 100.0) / 100.0 == 103.11);

    }
}
