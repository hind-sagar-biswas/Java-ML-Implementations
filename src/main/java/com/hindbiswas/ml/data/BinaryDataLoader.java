package com.hindbiswas.ml.data;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Utility to load Iris CSV and split into train/test sets.
 */
public class BinaryDataLoader {
    public static Map<String, Dataset> loadAndSplit(String csvPath, long seed, String label1, String label2)
            throws IOException, CsvValidationException {

        List<String[]> rows = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(csvPath))) {
            String[] line;
            reader.readNext(); // skip header
            while ((line = reader.readNext()) != null) {
                if (label1.equals(line[line.length - 1]) ||
                        label2.equals(line[line.length - 1])) {
                    rows.add(line);
                }
            }
        }

        Collections.shuffle(rows, new Random(seed));

        int splitIndex = (int) (rows.size() * 0.8);
        List<String[]> trainRows = rows.subList(0, splitIndex);
        List<String[]> testRows = rows.subList(splitIndex, rows.size());

        Dataset train = toDataset(trainRows, label1);
        Dataset test = toDataset(testRows, label1);

        Map<String, Dataset> map = new HashMap<>();
        map.put("train", train);
        map.put("test", test);
        return map;
    }

    private static Dataset toDataset(List<String[]> rows, String label1) {
        ArrayList<ArrayList<Double>> X = new ArrayList<>();
        ArrayList<Double> y = new ArrayList<>();

        for (String[] r : rows) {
            // Dynamically collect features
            ArrayList<Double> features = new ArrayList<>();
            for (int i = 0; i < r.length - 1; i++) {
                features.add(Double.parseDouble(r[i]));
            }
            X.add(features);

            // Encode last column as label
            String rawLabel = r[r.length - 1];
            double label = rawLabel.equals(label1) ? -1.0 : 1.0;
            y.add(label);
        }

        return new Dataset(X, y);
    }
}
