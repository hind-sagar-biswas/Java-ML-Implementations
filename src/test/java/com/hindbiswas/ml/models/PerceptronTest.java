package com.hindbiswas.ml.models;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the Perceptron classifier.
 */
public class PerceptronTest {

    private Perceptron perceptron;
    private ArrayList<ArrayList<Double>> dataX;
    private ArrayList<Double> dataY;

    @BeforeEach
    public void setUp() {
        perceptron = new Perceptron(0.1, 100, 0.0);
        dataX = new ArrayList<>();
        dataY = new ArrayList<>();
    }

    @Test
    public void testUnfittedToString() {
        // toString should indicate unfitted model before training
        Perceptron p = new Perceptron();
        assertEquals("Perceptron (unfitted)", p.toString());
    }

    @Test
    public void testPredictBeforeFitThrows() {
        // Predicting before fitting should throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            perceptron.predict(new ArrayList<>(Arrays.asList(1.0, 2.0)));
        });
    }

    @Test
    public void testEmptyDataThrows() {
        // Fitting on empty data should throw IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> {
            perceptron.fit(new ArrayList<>(), new ArrayList<>());
        });
    }

    @Test
    public void testMismatchedDataSizesThrows() {
        // dataX and dataY of different lengths should throw
        dataX.add(new ArrayList<>(Arrays.asList(0.0, 1.0)));
        assertThrows(IllegalArgumentException.class, () -> {
            perceptron.fit(dataX, dataY);
        });
    }

    @Test
    public void testInvalidLabelThrows() {
        // Labels other than -1 or +1 should throw
        dataX.add(new ArrayList<>(Arrays.asList(1.0, 1.0)));
        dataY.add(0.0);
        assertThrows(IllegalArgumentException.class, () -> {
            perceptron.fit(dataX, dataY);
        });
    }

    @Test
    public void testFitLinearlySeparableAND() {
        // AND logic: output +1 only when both inputs are 1, else -1
        dataX.add(new ArrayList<>(Arrays.asList(0.0, 0.0)));
        dataY.add(-1.0);
        dataX.add(new ArrayList<>(Arrays.asList(0.0, 1.0)));
        dataY.add(-1.0);
        dataX.add(new ArrayList<>(Arrays.asList(1.0, 0.0)));
        dataY.add(-1.0);
        dataX.add(new ArrayList<>(Arrays.asList(1.0, 1.0)));
        dataY.add(1.0);

        perceptron.fit(dataX, dataY);
        double score = perceptron.score(dataX, dataY);
        assertEquals(1.0, score, 1e-6, "Perceptron should perfectly classify AND dataset");
    }

    @Test
    public void testRandomSeedReproducibility() {
        // With zero iterations, initial weights come solely from seed
        // Use a trivial dataset of one point to allow fit()
        ArrayList<ArrayList<Double>> singleX = new ArrayList<>();
        ArrayList<Double> singleY = new ArrayList<>();
        singleX.add(new ArrayList<>(Arrays.asList(0.0)));
        singleY.add(1.0);

        Perceptron p1 = new Perceptron(0.1, 0, 0.0).randomizeWeights(42);
        Perceptron p2 = new Perceptron(0.1, 0, 0.0).randomizeWeights(42);
        p1.fit(singleX, singleY);
        p2.fit(singleX, singleY);

        assertEquals(p1.toString(), p2.toString(), "Models with same seed and zero iterations should match weights");
    }
}
