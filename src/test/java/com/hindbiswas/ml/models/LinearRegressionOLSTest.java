package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LinearRegressionOLSTest {
    private LinearRegressionOLS model;
    private ArrayList<Double> dataX;
    private ArrayList<Double> dataY;

    @BeforeEach
    void setUp() {
        model = LinearRegression.ols();
        dataX = new ArrayList<>();
        dataY = new ArrayList<>();
    }

    @Test
    void fitThrowsOnMismatchedSizes() {
        dataX.add(1.0);
        // dataY is empty
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void fitThrowsOnEmptyData() {
        // both empty
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void predictBeforeFitThrows() {
        assertThrows(IllegalStateException.class, () -> model.predict(5.0));
    }

    @Test
    void computesCorrectCoefficients() {
        // Known linear relationship: y = -1.75x + 103.106
        double[] xs = { 5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6 };
        double[] ys = { 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 };
        for (int i = 0; i < xs.length; i++) {
            dataX.add(xs[i]);
            dataY.add(ys[i]);
        }

        model.fit(dataX, dataY);
        double slope = model.getSlope();
        double intercept = model.getIntercept();

        // Round to two decimals for comparison
        assertEquals(-1.75, Math.round(slope * 100.0) / 100.0, 1e-9);
        assertEquals(103.11, Math.round(intercept * 100.0) / 100.0, 1e-9);
    }

    @Test
    void predictionMatchesLine() {
        // Use simple line y = 2x + 3
        double[][] df = {
                { 0.0, 1.0, 2.0, 3.0 },
                { 3.0, 5.0, 7.0, 9.0 }
        };

        for (int i = 0; i < df[0].length; i++) {
            dataX.add(df[0][i]);
            dataY.add(df[1][i]);
        }

        model.fit(new ArrayList<>(dataX), new ArrayList<>(dataY));

        // Expect slope ~2, intercept ~3
        assertEquals(2.0, model.getSlope(), 1e-9);
        assertEquals(3.0, model.getIntercept(), 1e-9);

        // Test predictions
        assertEquals(3.0, model.predict(0.0), 1e-9);
        assertEquals(5.0, model.predict(1.0), 1e-9);
        assertEquals(7.0, model.predict(2.0), 1e-9);
        assertEquals(9.0, model.predict(3.0), 1e-9);
    }

    @Test
    void toStringContainsCoefficients() {
        double[][] df = {
                { 1.0, 2.0 },
                { 2.0, 4.0 }
        };

        for (int i = 0; i < df[0].length; i++) {
            dataX.add(df[0][i]);
            dataY.add(df[1][i]);
        }

        model.fit(new ArrayList<>(dataX), new ArrayList<>(dataY));
        String repr = model.toString();
        assertTrue(repr.contains("y = ")); // basic format check
        assertTrue(repr.contains(model.getSlope().toString()));
        assertTrue(repr.contains(model.getIntercept().toString()));
    }
}
