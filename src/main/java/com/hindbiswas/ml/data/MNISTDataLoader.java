package com.hindbiswas.ml.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * MNISTDataLoader
 *
 * Expects CSV files where each row is:
 * label, px0, px1, ..., pxN
 * where label is an integer (0..9) and pixels are integers 0..255.
 *
 * The new overload `load(trainCsvPath, testCsvPath, limit)` reads up to `limit`
 * rows from each file. If limit <= 0, all rows are read.
 *
 * Returns a Map with keys "train" and "test" mapped to Dataset objects.
 */
public final class MNISTDataLoader {

    private MNISTDataLoader() {
        // utility class
    }

    /**
     * Load train and test CSV files and return a map containing "train" and "test"
     * Datasets. Loads up to `limit` rows from each file (limit <= 0 means "load
     * all").
     *
     * @param trainCsvPath path to training csv file
     * @param testCsvPath  path to testing csv file
     * @param limit        maximum number of rows to load from each CSV (<=0 => all
     *                     rows)
     * @return Map with keys "train" and "test"
     * @throws IOException              if file IO fails
     * @throws IllegalArgumentException if validation fails (missing files,
     *                                  inconsistent columns, bad numbers)
     */
    public static Map<String, Dataset> load(String trainCsvPath, String testCsvPath, int limit)
            throws IOException, IllegalArgumentException {
        Path trainPath = Paths.get(trainCsvPath);
        Path testPath = Paths.get(testCsvPath);

        validatePath(trainPath, "train");
        validatePath(testPath, "test");

        Dataset train = loadCsv(trainPath, limit);
        Dataset test = loadCsv(testPath, limit);

        Map<String, Dataset> result = new HashMap<>();
        result.put("train", train);
        result.put("test", test);
        return result;
    }

    /**
     * Load train and test CSV files and return a map containing "train" and "test"
     * Datasets. Loads up to `limit` rows from each file (limit <= 0 means "load
     * all").
     *
     * @param trainCsvPath path to training csv file
     * @param trainLimit   maximum number of rows to load from train CSV (<=0 => all
     * @param testCsvPath  path to testing csv file
     * @param testLimit    maximum number of rows to load from test CSV (<=0 => all
     *                     rows)
     * @return Map with keys "train" and "test"
     * @throws IOException              if file IO fails
     * @throws IllegalArgumentException if validation fails (missing files,
     *                                  inconsistent columns, bad numbers)
     */
    public static Map<String, Dataset> load(String trainCsvPath, int trainLimit, String testCsvPath, int testLimit)
            throws IOException, IllegalArgumentException {
        Path trainPath = Paths.get(trainCsvPath);
        Path testPath = Paths.get(testCsvPath);

        validatePath(trainPath, "train");
        validatePath(testPath, "test");

        Dataset train = loadCsv(trainPath, trainLimit);
        Dataset test = loadCsv(testPath, testLimit);

        Map<String, Dataset> result = new HashMap<>();
        result.put("train", train);
        result.put("test", test);
        return result;
    }

    /** Backwards-compatible convenience: load all rows from both files. */
    public static Map<String, Dataset> load(String trainCsvPath, String testCsvPath)
            throws IOException, IllegalArgumentException {
        return load(trainCsvPath, testCsvPath, 0);
    }

    private static void validatePath(Path p, String name) {
        if (!Files.exists(p)) {
            throw new IllegalArgumentException("The " + name + " file does not exist: " + p.toString());
        }
        if (!Files.isRegularFile(p) || !Files.isReadable(p)) {
            throw new IllegalArgumentException("The " + name + " path is not a readable file: " + p.toString());
        }
    }

    /**
     * Loads up to `limit` rows from path. If limit <= 0 -> loads all rows.
     */
    private static Dataset loadCsv(Path path, int limit) throws IOException, IllegalArgumentException {
        ArrayList<ArrayList<Double>> X = new ArrayList<>();
        ArrayList<Double> y = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            boolean headerChecked = false;
            int expectedCols = -1;
            int loaded = 0;

            while ((line = br.readLine()) != null) {
                if (limit > 0 && loaded >= limit)
                    break; // stop when we've loaded enough

                line = line.trim();
                if (line.isEmpty())
                    continue; // skip blank lines

                String[] rawTokens = line.split(",");
                // trim tokens
                String[] tokens = new String[rawTokens.length];
                for (int i = 0; i < rawTokens.length; i++)
                    tokens[i] = rawTokens[i].trim();

                // Detect and skip header (if first token is not a number)
                if (!headerChecked) {
                    headerChecked = true;
                    try {
                        Integer.parseInt(tokens[0]);
                    } catch (NumberFormatException ex) {
                        // header detected -> skip this line
                        continue;
                    }
                    expectedCols = tokens.length;
                }

                if (expectedCols == -1) {
                    expectedCols = tokens.length;
                }

                if (tokens.length != expectedCols) {
                    throw new IllegalArgumentException("Inconsistent number of columns in CSV " + path.toString()
                            + ". Expected " + expectedCols + " but got " + tokens.length + " in line: " + line);
                }

                // parse label (first column)
                int label;
                try {
                    label = Integer.parseInt(tokens[0]);
                } catch (NumberFormatException ex) {
                    throw new IllegalArgumentException(
                            "Invalid label value in file " + path.toString() + " : " + tokens[0]);
                }
                y.add((double) label);

                // parse pixels and normalize to [0,1]
                ArrayList<Double> pixels = new ArrayList<>(tokens.length - 1);
                for (int i = 1; i < tokens.length; i++) {
                    try {
                        double px = Double.parseDouble(tokens[i]);
                        if (px < 0 || px > 255) {
                            throw new NumberFormatException();
                        }
                        pixels.add(px / 255.0);
                    } catch (NumberFormatException ex) {
                        throw new IllegalArgumentException(
                                "Invalid pixel value in file " + path.toString() + " : " + tokens[i]);
                    }
                }
                X.add(pixels);
                loaded++;
            }
        }

        if (X.isEmpty()) {
            throw new IllegalArgumentException("No data found in file: " + path.toString());
        }

        // optional sanity check: most MNIST CSVs are 784 pixels (28*28)
        int pixelCount = X.get(0).size();
        if (pixelCount != 784) {
            System.out.printf("Warning: loaded CSV %s has %d pixels per row (expected 784).%n", path.toString(),
                    pixelCount);
        }

        return new Dataset(X, y);
    }
}
