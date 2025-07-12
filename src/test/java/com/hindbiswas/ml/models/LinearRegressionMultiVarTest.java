package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LinearRegressionMultiVarTest {
    private LinearRegressionMultiVar model;
    private ArrayList<ArrayList<Double>> dataX;
    private ArrayList<Double> dataY;

    @BeforeEach
    void setUp() {
        model = LinearRegression.multi();
        dataX = new ArrayList<>();
        dataY = new ArrayList<>();
    }

    @Test
    void fitThrowsOnMismatchedSizes() {
        dataX.add(new ArrayList<>(Arrays.asList(1.0, 2.0)));
        // dataY empty
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void fitThrowsOnEmptyData() {
        // both empty
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void fitThrowsOnEmptyFeatureVector() {
        dataX.add(new ArrayList<>()); // empty feature list
        dataY.add(1.0);
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void predictBeforeFitThrows() {
        ArrayList<Double> features = new ArrayList<>(Arrays.asList(1.0));
        assertThrows(IllegalStateException.class, () -> model.predict(features));
    }

    @Test
    void computesCorrectCoefficientsSingleFeature() {
        // y = 2x + 3
        for (double xv : Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)) {
            dataX.add(new ArrayList<>(Collections.singletonList(xv)));
            dataY.add(2 * xv + 3);
        }

        model.fit(dataX, dataY);
        SimpleMatrix theta = model.getTheta();
        double intercept = theta.get(0, 0);
        double slope = theta.get(1, 0);

        assertEquals(3.0, intercept, 1e-9);
        assertEquals(2.0, slope, 1e-9);
    }

    @Test
    void predictionMatchesLine() {
        // y = -1.75x + 103.106
        double[] xs = { 5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6 };
        double[] ys = { 99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 };
        for (int i = 0; i < xs.length; i++) {
            dataX.add(new ArrayList<>(Collections.singletonList(xs[i])));
            dataY.add(ys[i]);
        }

        model.fit(dataX, dataY);
        ArrayList<Double> newX = new ArrayList<>(Collections.singletonList(10.0));
        double pred = model.predict(newX);
        // approximate line: y = -1.75*10 + 103.106 = 85.606
        assertEquals(-1.75, Math.round(model.getTheta().get(1, 0) * 100.0) / 100.0, 1e-9);
        assertEquals(103.11, Math.round(model.getTheta().get(0, 0) * 100.0) / 100.0, 1e-9);
        assertEquals(85.59, Math.round(pred * 100.0) / 100.0, 1e-9);
    }

    @Test
    void toStringContainsCoefficients() {
        // simple y = x + 1
        dataX.add(new ArrayList<>(Arrays.asList(1.0)));
        dataX.add(new ArrayList<>(Arrays.asList(2.0)));
        dataY.add(2.0);
        dataY.add(3.0);
        model.fit(dataX, dataY);

        String repr = model.toString();
        assertTrue(repr.contains("y ="));
        SimpleMatrix theta = model.getTheta();
        assertTrue(repr.contains(String.format("%.4f", theta.get(0, 0))));
        assertTrue(repr.contains(String.format("%.4f", theta.get(1, 0))));
    }

    @Test
    void getThetaBeforeFitThrows() {
        assertThrows(IllegalStateException.class, () -> model.getTheta());
    }
}
