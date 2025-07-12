package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LinearRegressionGDTest {
    private LinearRegressionGD model;
    private ArrayList<ArrayList<Double>> dataX;
    private ArrayList<Double> dataY;

    @BeforeEach
    void setUp() {
        // Use a small learning rate and enough iterations for convergence
        model = LinearRegression.gradient(0.01, 10000);
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
        dataX.add(new ArrayList<>());
        dataY.add(1.0);
        assertThrows(IllegalArgumentException.class, () -> model.fit(dataX, dataY));
    }

    @Test
    void predictBeforeFitThrows() {
        ArrayList<Double> features = new ArrayList<>(Arrays.asList(1.0));
        assertThrows(IllegalStateException.class, () -> model.predict(features));
    }

    @Test
    void learnsSimpleLinearFunction() {
        // y = 2x + 3
        for (double xv : Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)) {
            dataX.add(new ArrayList<>(Collections.singletonList(xv)));
            dataY.add(2 * xv + 3);
        }

        model.fit(dataX, dataY);
        SimpleMatrix theta = model.getTheta();
        double intercept = theta.get(0, 0);
        double slope = theta.get(1, 0);

        // Allow slight tolerance due to gradient descent
        assertEquals(3.0, intercept, 1e-2);
        assertEquals(2.0, slope, 1e-2);

        // Test prediction
        double pred = model.predict(new ArrayList<>(Collections.singletonList(10.0)));
        assertEquals(2 * 10 + 3, pred, 1e-1);
    }

    @Test
    void learnsMultivariateFunction() {
        // y = 1*x1 + 2*x2 + 1
        for (double x1 : Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0)) {
            for (double x2 : Arrays.asList(2.0, 1.0, 0.0, -1.0, -2.0)) {
                dataX.add(new ArrayList<>(Arrays.asList(x1, x2)));
                dataY.add(1 * x1 + 2 * x2 + 1);
            }
        }

        model.fit(dataX, dataY);
        SimpleMatrix theta = model.getTheta();
        // theta: [intercept; coef1; coef2]
        assertEquals(1.0, theta.get(0, 0), 1e-1);
        assertEquals(1.0, theta.get(1, 0), 1e-1);
        assertEquals(2.0, theta.get(2, 0), 1e-1);
    }

    @Test
    void toStringContainsThetaValues() {
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
        assertThrows(IllegalStateException.class, () -> model.getTheta().get(0, 0));
    }
}
