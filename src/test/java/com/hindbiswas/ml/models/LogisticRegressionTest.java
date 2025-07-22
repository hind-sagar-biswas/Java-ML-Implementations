package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LogisticRegressionTest {
    private LogisticRegression model;

    @BeforeEach
    void setUp() {
        model = new LogisticRegression(0.1, 1000);
    }

    @Test
    void testFitThrowsOnMismatchedSizes() {
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        xs.add(new ArrayList<>());
        ArrayList<Double> ys = new ArrayList<>();
        assertThrows(IllegalArgumentException.class, () -> model.fit(xs, ys));
    }

    @Test
    void testFitThrowsOnEmptyFeatures() {
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        xs.add(new ArrayList<>());
        ArrayList<Double> ys = new ArrayList<>();
        ys.add(0.0);
        assertThrows(IllegalArgumentException.class, () -> model.fit(xs, ys));
    }

    @Test
    void testFitThrowsOnInvalidY() {
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        ArrayList<Double> ys = new ArrayList<>();
        ArrayList<Double> row = new ArrayList<>();
        row.add(1.0);
        xs.add(row);
        ys.add(2.0);
        assertThrows(IllegalArgumentException.class, () -> model.fit(xs, ys));
    }

    @Test
    void testPredictBeforeFitThrows() {
        ArrayList<Double> x = new ArrayList<>();
        x.add(1.0);
        assertThrows(IllegalStateException.class, () -> model.classify(x));
    }

    @Test
    void testSimpleLineSeparable() {
        // Create a dataset where y = 1 if x > 0.5, else 0
        ArrayList<ArrayList<Double>> xs = new ArrayList<>();
        ArrayList<Double> ys = new ArrayList<>();
        for (double v : new double[] { 0.1, 0.2, 0.8, 0.9 }) {
            ArrayList<Double> row = new ArrayList<>();
            row.add(v);
            xs.add(row);
            ys.add(v > 0.5 ? 1.0 : 0.0);
        }
        model = new LogisticRegression(0.5, 5000);
        model.fit(xs, ys);

        // theta[1] should be positive and theta[0] negative roughly
        SimpleMatrix theta = model.getTheta();
        assertTrue(theta.get(1, 0) > 0);
        assertTrue(theta.get(0, 0) < 0);

        // Test predictions
        ArrayList<Double> testLow = new ArrayList<>();
        testLow.add(0.2);
        ArrayList<Double> testHigh = new ArrayList<>();
        testHigh.add(0.8);
        assertEquals(0, model.classify(testLow));
        assertEquals(1, model.classify(testHigh));
    }
}
