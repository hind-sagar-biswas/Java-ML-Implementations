package com.hindbiswas.ml;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.Test;

import com.hindbiswas.ml.models.LinearRegressionOLS;

/**
 * Unit test for simple App.
 */
public class LinearRegressionOLSTest {
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue() {
        ArrayList<Double> dataX = new ArrayList<Double>();
        ArrayList<Double> dataY = new ArrayList<Double>();

        double[][] data = {
                { 5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6 },
                { 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 }
        };

        for (int i = 0; i < data[0].length; i++) {
            dataX.add(data[0][i]);
            dataY.add(data[1][i]);
        }

        LinearRegressionOLS model = new LinearRegressionOLS();
        model.fit(dataX, dataY);

        double m = model.getSlope();
        double b = model.getIntercept();

        // LinearRegression [y = -1.7512877115526122x + 103.10596026490066]
        assertTrue(Math.round(m * 100.0) / 100.0 == -1.75);
        assertTrue(Math.round(b * 100.0) / 100.0 == 103.11);
    }
}
