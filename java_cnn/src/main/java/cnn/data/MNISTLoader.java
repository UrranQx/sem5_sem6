package cnn.data;

import cnn.utils.Tensor3D;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * MNIST dataset loader
 * Can load from local files or generate synthetic MNIST-like data
 * <p>
 * To use real MNIST data, download the following files to mnist_data/ directory:
 * - train-images-idx3-ubyte.gz
 * - train-labels-idx1-ubyte.gz
 * - t10k-images-idx3-ubyte.gz
 * - t10k-labels-idx1-ubyte.gz
 * or if you have emnist (extended MNIST) dataset
 * - emnist-digits-train-images-idx3-ubyte.gz
 * - emnist-digits-train-train-labels-idx1-ubyte.gz
 * - emnist-digits-test-images-idx3-ubyte.gz
 * - emnist-digits-test-labels-idx1-ubyte.gz
 * or there is smaller version of MNIST files
 * Files can be obtained from: http://yann.lecun.com/exdb/mnist/
 */
public class MNISTLoader {
    private static final int USE_SMALLER_MNIST = 1; // 0 - full EMNIST digits, 1 - smaller MNIST-like EMNIST
    private static final String[][] FILE_NAMES = {
            // Digits: 280,000 characters, 10 balanced classes (0-9).
            {       "emnist-digits-train-images-idx3-ubyte.gz",
                    "emnist-digits-train-labels-idx1-ubyte.gz",
                    "emnist-digits-test-images-idx3-ubyte.gz",
                    "emnist-digits-test-labels-idx1-ubyte.gz"},
            {
                    "emnist-mnist-train-images-idx3-ubyte.gz",
                    "emnist-mnist-train-labels-idx1-ubyte.gz",
                    "emnist-mnist-test-images-idx3-ubyte.gz",
                    "emnist-mnist-test-labels-idx1-ubyte.gz"
            }
    };
    /* Alternative MNIST file names, they are smaller, like MNIST: 70,000 characters, 10 balanced classes (directly compatible with MNIST).

        "emnist-mnist-train-images-idx3-ubyte.gz",
        "emnist-mnist-train-labels-idx1-ubyte.gz",
        "emnist-mnist-test-images-idx3-ubyte.gz",
        "emnist-mnist-test-labels-idx1-ubyte.gz"

     */

    // Digit patterns for synthetic data generation (simple 7x7 patterns scaled to 28x28)
    private static final int[][][] DIGIT_PATTERNS = {
            // 0
            {{0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}},
            // 1
            {{0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}},
            // 2
            {{0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {0, 0, 0, 1, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 1, 1, 1, 1}},
            // 3
            {{0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {0, 0, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}},
            // 4
            {{0, 0, 0, 1, 0}, {0, 0, 1, 1, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 1, 0}, {1, 1, 1, 1, 1}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 0}},
            // 5
            {{1, 1, 1, 1, 1}, {1, 0, 0, 0, 0}, {1, 1, 1, 1, 0}, {0, 0, 0, 0, 1}, {0, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}},
            // 6
            {{0, 0, 1, 1, 0}, {0, 1, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}},
            // 7
            {{1, 1, 1, 1, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}},
            // 8
            {{0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 0}},
            // 9
            {{0, 1, 1, 1, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {0, 1, 1, 1, 1}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, 0}, {0, 1, 1, 0, 0}}
    };

    private float[][][] trainImages;
    private int[] trainLabels;
    private float[][][] testImages;
    private int[] testLabels;

    private String dataDir;
    private boolean useSyntheticData;

    public MNISTLoader(String dataDir) {
        this.dataDir = dataDir;
        this.useSyntheticData = false;
    }

    /**
     * Check if all MNIST files exist locally
     */
    private boolean filesExist() {
        Path dirPath = Paths.get(dataDir);
        System.out.println("Checking for MNIST files in directory: " + dirPath.toAbsolutePath());
        for (String fileName : FILE_NAMES[USE_SMALLER_MNIST]) {
            if (!Files.exists(dirPath.resolve(fileName))) {
                System.out.println("File not found: " + fileName);
                System.out.println("Expected path: " + dirPath.resolve(fileName).toAbsolutePath());
                return false;
            }
        }
        return true;
    }

    /**
     * Load MNIST dataset from files or generate synthetic data
     */
    public void load() throws IOException {
        if (filesExist()) {
            System.out.println("Loading MNIST data from local files...");
            loadFromFiles();
        } else {

            System.out.println("MNIST files not found. Generating synthetic data...");
            System.out.println("(For real MNIST data, download files to " + dataDir + "/)");
            generateSyntheticData();
            useSyntheticData = true;
        }
    }

    /**
     * Load from local MNIST files
     */
    private void loadFromFiles() throws IOException {
        System.out.println("Loading MNIST training images...");
        trainImages = loadImages(Paths.get(dataDir, FILE_NAMES[USE_SMALLER_MNIST][0]).toString());
        System.out.println("Loading MNIST training labels...");
        trainLabels = loadLabels(Paths.get(dataDir, FILE_NAMES[USE_SMALLER_MNIST][1]).toString());
        System.out.println("Loading MNIST test images...");
        testImages = loadImages(Paths.get(dataDir, FILE_NAMES[USE_SMALLER_MNIST][2]).toString());
        System.out.println("Loading MNIST test labels...");
        testLabels = loadLabels(Paths.get(dataDir, FILE_NAMES[USE_SMALLER_MNIST][3]).toString());

        System.out.println("Loaded " + trainImages.length + " training images and " + testImages.length + " test images");
    }

    /**
     * Generate synthetic MNIST-like data for testing
     */
    private void generateSyntheticData() {
        Random rand = new Random(42);

        int trainSize = 6000;  // Smaller for testing
        int testSize = 1000;

        trainImages = new float[trainSize][28][28];
        trainLabels = new int[trainSize];
        testImages = new float[testSize][28][28];
        testLabels = new int[testSize];

        // Generate training data
        for (int i = 0; i < trainSize; i++) {
            int digit = rand.nextInt(10);
            trainLabels[i] = digit;
            trainImages[i] = generateDigitImage(digit, rand);
        }

        // Generate test data
        for (int i = 0; i < testSize; i++) {
            int digit = rand.nextInt(10);
            testLabels[i] = digit;
            testImages[i] = generateDigitImage(digit, rand);
        }

        System.out.println("Generated " + trainSize + " training images and " + testSize + " test images");
    }

    /**
     * Generate a 28x28 image of a digit with random variations
     */
    private float[][] generateDigitImage(int digit, Random rand) {
        float[][] image = new float[28][28];
        int[][] pattern = DIGIT_PATTERNS[digit];

        // Random offset for position variation (-2 to +2 from center position)
        // Base offsets: X=2 for horizontal centering, Y=4 for vertical centering
        int offsetX = 2 + (rand.nextInt(5) - 2);  // Range: 0-4
        int offsetY = 4 + (rand.nextInt(5) - 2);  // Range: 2-6

        // Scale factor (pattern is 5x7, scale to ~20x21)
        int scaleX = 4;
        int scaleY = 3;

        for (int py = 0; py < pattern.length; py++) {
            for (int px = 0; px < pattern[0].length; px++) {
                if (pattern[py][px] == 1) {
                    // Fill scaled pixels
                    for (int sy = 0; sy < scaleY; sy++) {
                        for (int sx = 0; sx < scaleX; sx++) {
                            int imgY = offsetY + py * scaleY + sy;
                            int imgX = offsetX + px * scaleX + sx;
                            if (imgX >= 0 && imgX < 28 && imgY >= 0 && imgY < 28) {
                                // Add some noise variation
                                image[imgY][imgX] = 0.7f + rand.nextFloat() * 0.3f;
                            }
                        }
                    }
                }
            }
        }

        // Add some random noise
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                if (image[y][x] == 0 && rand.nextFloat() < 0.02f) {
                    image[y][x] = rand.nextFloat() * 0.3f;
                }
            }
        }

        return image;
    }

    /**
     * Check if using synthetic data
     */
    public boolean isUsingSyntheticData() {
        return useSyntheticData;
    }

    /**
     * Load images from IDX file format
     */
    private float[][][] loadImages(String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new GZIPInputStream(new FileInputStream(filename)))) {

            int magic = dis.readInt();
            if (magic != 2051) {
                throw new IOException("Invalid magic number for images: " + magic);
            }

            int numImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

            float[][][] images = new float[numImages][numRows][numCols];

            for (int i = 0; i < numImages; i++) {
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        // Normalize to [0, 1]
                        images[i][r][c] = (dis.readUnsignedByte() & 0xFF) / 255.0f;
                    }
                }
            }

            return images;
        }
    }

    /**
     * Load labels from IDX file format
     */
    private int[] loadLabels(String filename) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new GZIPInputStream(new FileInputStream(filename)))) {

            int magic = dis.readInt();
            if (magic != 2049) {
                throw new IOException("Invalid magic number for labels: " + magic);
            }

            int numLabels = dis.readInt();
            int[] labels = new int[numLabels];

            for (int i = 0; i < numLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }

            return labels;
        }
    }

    /**
     * Get training data split by ratio (e.g., 0.7 for 70% training)
     */
    public DataSplit getTrainValidationSplit(double trainRatio, long seed) {
        Random rand = new Random(seed);

        // Combine train and test into one pool
        int totalSize = trainImages.length + testImages.length;
        float[][][] allImages = new float[totalSize][][];
        int[] allLabels = new int[totalSize];

        System.arraycopy(trainImages, 0, allImages, 0, trainImages.length);
        System.arraycopy(testImages, 0, allImages, trainImages.length, testImages.length);
        System.arraycopy(trainLabels, 0, allLabels, 0, trainLabels.length);
        System.arraycopy(testLabels, 0, allLabels, trainLabels.length, testLabels.length);

        // Shuffle indices
        Integer[] indices = new Integer[totalSize];
        for (int i = 0; i < totalSize; i++) indices[i] = i;
        Collections.shuffle(Arrays.asList(indices), rand);

        // Split
        int trainSize = (int) (totalSize * trainRatio);
        int testSize = totalSize - trainSize;

        Tensor3D[] trainX = new Tensor3D[trainSize];
        float[][] trainY = new float[trainSize][10];
        Tensor3D[] testX = new Tensor3D[testSize];
        float[][] testY = new float[testSize][10];
        int[] testLabelsList = new int[testSize];
        int[] trainLabelsList = new int[trainSize];

        for (int i = 0; i < trainSize; i++) {
            int idx = indices[i];
            trainX[i] = Tensor3D.fromArray2D(allImages[idx]);
            trainY[i] = oneHot(allLabels[idx], 10);
            trainLabelsList[i] = allLabels[idx];
        }

        for (int i = 0; i < testSize; i++) {
            int idx = indices[trainSize + i];
            testX[i] = Tensor3D.fromArray2D(allImages[idx]);
            testY[i] = oneHot(allLabels[idx], 10);
            testLabelsList[i] = allLabels[idx];
        }

        return new DataSplit(trainX, trainY, trainLabelsList, testX, testY, testLabelsList);
    }

    /**
     * One-hot encode label
     */
    private float[] oneHot(int label, int numClasses) {
        float[] result = new float[numClasses];
        result[label] = 1.0f;
        return result;
    }

    /**
     * Data split holder
     */
    public static class DataSplit {
        public final Tensor3D[] trainX;
        public final float[][] trainY;
        public final int[] trainLabels;
        public final Tensor3D[] testX;
        public final float[][] testY;
        public final int[] testLabels;

        public DataSplit(Tensor3D[] trainX, float[][] trainY, int[] trainLabels,
                         Tensor3D[] testX, float[][] testY, int[] testLabels) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.trainLabels = trainLabels;
            this.testX = testX;
            this.testY = testY;
            this.testLabels = testLabels;
        }
    }

    /**
     * Print ASCII representation of an image
     */
    public static void printImage(Tensor3D image, int label, int predicted) {
        System.out.println("Label: " + label + " | Predicted: " + predicted);

        // Получаем данные изображения (первый канал)
        float[][] data = image.data[0];

        // Транспонируем изображение
        float[][] transposedData = transposeMatrix(data);

        // Печать строки, представляющей изображение
        for (int r = 0; r < transposedData.length; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < transposedData[r].length; c++) {
                float val = transposedData[r][c];
                if (val > 0.75f) sb.append("██");
                else if (val > 0.5f) sb.append("▓▓");
                else if (val > 0.25f) sb.append("░░");
                else sb.append("  ");
            }
            System.out.println(sb);
        }
        System.out.println();
    }

    /**
     * Transpose a 2D matrix (used for image data)
     *
     * @param matrix The matrix to be transposed
     * @return The transposed matrix
     */
    private static float[][] transposeMatrix(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] transposed = new float[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
}
