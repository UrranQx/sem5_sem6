package cnn;

import cnn.data.MNISTLoader;
import cnn.data.MNISTLoader.DataSplit;
import cnn.layers.*;
import cnn.utils.ConfusionMatrix;
import cnn.utils.Tensor3D;

import java.io.IOException;
import java.util.Random;
import java.io.File;
/**
 * Main class for CNN MNIST classification
 * 
 * Network architecture:
 * - Input: 28x28x1 (MNIST image)
 * - Conv2D: 8 filters, 3x3, padding=1 -> 28x28x8
 * - ReLU
 * - MaxPool: 2x2 -> 14x14x8
 * - Conv2D: 16 filters, 3x3, padding=1 -> 14x14x16
 * - ReLU
 * - MaxPool: 2x2 -> 7x7x16
 * - Flatten -> 784
 * - Dense: 128
 * - ReLU
 * - Dense: 10
 * - Softmax
 */
public class Main {
    
    // Hyperparameters
    private static final float LEARNING_RATE = 0.001f;
    private static final int EPOCHS = 3;
    private static final double TRAIN_RATIO = 0.7;
    private static final long SEED = 42;
    private static final int SAMPLE_DISPLAY_COUNT = 10;
    
    public static void main(String[] args) {
        System.out.println("=" .repeat(60));
        System.out.println("CNN for MNIST Classification using Java Vector API");
        System.out.println("=" .repeat(60));
        
        try {
            // Load MNIST data
            System.out.println("\n[1] Loading MNIST dataset...");

            System.out.println("Текущая рабочая директория: " + System.getProperty("user.dir"));

            // Попробуйте, например, проверить, существует ли папка mnist_data в текущей директории
            File mnistFolder = new File(System.getProperty("user.dir") + File.separator + "mnist_data");
            if (mnistFolder.exists() && mnistFolder.isDirectory()) {
                System.out.println("Папка найдена!");
            } else {
                System.out.println("Папка mnist_data НЕ найдена в текущей директории.");
            }

            // MNISTLoader loader = new MNISTLoader("mnist_data");
            MNISTLoader loader = new MNISTLoader(System.getProperty("user.dir") + File.separator + "mnist_data");
            loader.load();
            
            // Split data 70/30
            System.out.println("\n[2] Splitting data (70% train, 30% test)...");
            DataSplit data = loader.getTrainValidationSplit(TRAIN_RATIO, SEED);
            System.out.println("Training samples: " + data.trainX.length);
            System.out.println("Test samples: " + data.testX.length);
            
            // Build CNN
            System.out.println("\n[3] Building CNN architecture...");
            CNN cnn = buildNetwork();
            cnn.summary();
            
            // Training
            System.out.println("\n[4] Training the network...");
            trainNetwork(cnn, data);
            
            // Testing
            System.out.println("\n[5] Evaluating on test set...");
            ConfusionMatrix confMatrix = evaluateNetwork(cnn, data);
            confMatrix.print();
            
            // Display samples
            System.out.println("\n[6] Displaying sample predictions...");
            displaySamples(cnn, data, SAMPLE_DISPLAY_COUNT);
            
            System.out.println("\n" + "=" .repeat(60));
            System.out.println("Done!");
            System.out.println("=" .repeat(60));
            
        } catch (IOException e) {
            System.err.println("Error loading data: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Build the CNN architecture
     */
    private static CNN buildNetwork() {
        CNN cnn = new CNN(LEARNING_RATE);
        
        // First conv block
        cnn.addLayer(new ConvLayer(1, 8, 3, 1, 1));  // 28x28x1 -> 28x28x8
        cnn.addLayer(new ReLULayer());
        cnn.addLayer(new MaxPoolLayer(2));           // 28x28x8 -> 14x14x8
        
        // Second conv block
        cnn.addLayer(new ConvLayer(8, 16, 3, 1, 1)); // 14x14x8 -> 14x14x16
        cnn.addLayer(new ReLULayer());
        cnn.addLayer(new MaxPoolLayer(2));           // 14x14x16 -> 7x7x16
        
        // Fully connected layers
        cnn.addLayer(new FlattenLayer());            // 7x7x16 = 784
        cnn.addLayer(new DenseLayer(7 * 7 * 16, 128));
        cnn.addLayer(new ReLULayer());
        cnn.addLayer(new DenseLayer(128, 10));
        cnn.addLayer(new SoftmaxLayer());
        
        return cnn;
    }
    
    /**
     * Train the network
     */
    private static void trainNetwork(CNN cnn, DataSplit data) {
        Random rand = new Random(SEED);
        int trainSize = data.trainX.length;
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            float totalLoss = 0;
            int correct = 0;
            
            // Shuffle training indices
            int[] indices = new int[trainSize];
            for (int i = 0; i < trainSize; i++) indices[i] = i;
            shuffleArray(indices, rand);
            
            long startTime = System.currentTimeMillis();
            
            for (int i = 0; i < trainSize; i++) {
                int idx = indices[i];
                
                // Forward and backward pass
                float loss = cnn.trainStep(data.trainX[idx], data.trainY[idx]);
                totalLoss += loss;
                
                // Check prediction
                int predicted = cnn.predict(data.trainX[idx]);
                if (predicted == data.trainLabels[idx]) {
                    correct++;
                }
                
                // Progress update
                if ((i + 1) % 1000 == 0 || i == trainSize - 1) {
                    float avgLoss = totalLoss / (i + 1);
                    float accuracy = (float) correct / (i + 1) * 100;
                    System.out.printf("\rEpoch %d/%d: [%d/%d] Loss: %.4f, Accuracy: %.2f%%", 
                        epoch + 1, EPOCHS, i + 1, trainSize, avgLoss, accuracy);
                }
            }
            
            long endTime = System.currentTimeMillis();
            float epochLoss = totalLoss / trainSize;
            float epochAcc = (float) correct / trainSize * 100;
            
            System.out.printf("\rEpoch %d/%d completed in %.1fs - Loss: %.4f, Accuracy: %.2f%%%n", 
                epoch + 1, EPOCHS, (endTime - startTime) / 1000.0, epochLoss, epochAcc);
        }
    }
    
    /**
     * Evaluate the network on test data
     */
    private static ConfusionMatrix evaluateNetwork(CNN cnn, DataSplit data) {
        ConfusionMatrix confMatrix = new ConfusionMatrix(10);
        
        int testSize = data.testX.length;
        for (int i = 0; i < testSize; i++) {
            int predicted = cnn.predict(data.testX[i]);
            int actual = data.testLabels[i];
            confMatrix.add(actual, predicted);
            
            if ((i + 1) % 1000 == 0) {
                System.out.printf("\rEvaluating: [%d/%d]", i + 1, testSize);
            }
        }
        System.out.println();
        
        return confMatrix;
    }
    
    /**
     * Display sample predictions
     */
    private static void displaySamples(CNN cnn, DataSplit data, int count) {
        Random rand = new Random(SEED + 1);
        int testSize = data.testX.length;
        
        System.out.println("\n" + "=" .repeat(60));
        System.out.println("SAMPLE PREDICTIONS (" + count + " random test images)");
        System.out.println("=" .repeat(60));
        
        for (int i = 0; i < count; i++) {
            int idx = rand.nextInt(testSize);
            int predicted = cnn.predict(data.testX[idx]);
            int actual = data.testLabels[idx];
            
            float[] probs = cnn.predictProba(data.testX[idx]);
            
            System.out.println("\n--- Sample " + (i + 1) + " ---");
            MNISTLoader.printImage(data.testX[idx], actual, predicted);
            
            System.out.print("Probabilities: ");
            for (int j = 0; j < 10; j++) {
                System.out.printf("%d:%.2f ", j, probs[j]);
            }
            System.out.println();
            
            if (predicted == actual) {
                System.out.println("✓ CORRECT");
            } else {
                System.out.println("✗ INCORRECT (Expected: " + actual + ", Got: " + predicted + ")");
            }
        }
    }
    
    /**
     * Shuffle array using Fisher-Yates algorithm
     */
    private static void shuffleArray(int[] arr, Random rand) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
}
