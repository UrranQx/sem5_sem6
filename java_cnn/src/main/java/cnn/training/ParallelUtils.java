package cnn.training;

import cnn.CNN;
import cnn.utils.Tensor3D;
import cnn.utils.ConfusionMatrix;

import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

/**
 * Parallel utilities for CNN operations
 * 
 * Provides multi-threaded implementations for:
 * - Parallel prediction/inference
 * - Parallel evaluation with confusion matrix
 * - Batch prediction with thread pool
 * 
 * Note: Training is kept sequential because CNN layers maintain internal state
 * (lastInput, lastOutput) for backpropagation. For true parallel training,
 * you would need to clone the network for each thread or use gradient accumulation
 * with proper synchronization.
 */
public class ParallelUtils {
    
    private static final int DEFAULT_PARALLELISM = Runtime.getRuntime().availableProcessors();
    
    /**
     * Parallel prediction using ForkJoinPool
     * Each prediction is independent, so this is perfectly parallelizable
     * 
     * @param cnn The CNN model (must be thread-safe for forward pass only)
     * @param inputs Array of input tensors
     * @return Array of predicted class labels
     */
    public static int[] parallelPredict(CNN cnn, Tensor3D[] inputs) {
        int[] predictions = new int[inputs.length];
        
        // Use parallel streams for simple parallelization
        // Note: CNN forward pass may not be thread-safe due to internal state
        // This works for inference if we don't need backward pass
        IntStream.range(0, inputs.length)
            .parallel()
            .forEach(i -> {
                synchronized (cnn) {
                    predictions[i] = cnn.predict(inputs[i]);
                }
            });
        
        return predictions;
    }
    
    /**
     * Parallel evaluation with confusion matrix building
     * 
     * @param cnn The CNN model
     * @param testX Test images
     * @param testLabels True labels
     * @param numClasses Number of classes (10 for MNIST)
     * @return ConfusionMatrix with all predictions
     */
    public static ConfusionMatrix parallelEvaluate(CNN cnn, Tensor3D[] testX, int[] testLabels, int numClasses) {
        ConfusionMatrix confMatrix = new ConfusionMatrix(numClasses);
        int testSize = testX.length;
        
        // Collect predictions in parallel
        int[] predictions = parallelPredict(cnn, testX);
        
        // Build confusion matrix (sequential, fast)
        for (int i = 0; i < testSize; i++) {
            confMatrix.add(testLabels[i], predictions[i]);
        }
        
        return confMatrix;
    }
    
    /**
     * Calculate accuracy in parallel using streams
     * 
     * @param predictions Predicted labels
     * @param labels True labels
     * @return Accuracy percentage (0-100)
     */
    public static float parallelAccuracy(int[] predictions, int[] labels) {
        long correct = IntStream.range(0, predictions.length)
            .parallel()
            .filter(i -> predictions[i] == labels[i])
            .count();
        
        return (float) correct / predictions.length * 100;
    }
    
    /**
     * Chunked parallel prediction with explicit thread pool
     * Better for fine-grained control and when CNN is not thread-safe
     * 
     * @param cnn The CNN model
     * @param inputs Input tensors
     * @param numThreads Number of worker threads
     * @return Predicted labels
     */
    public static int[] chunkedParallelPredict(CNN cnn, Tensor3D[] inputs, int numThreads) {
        int testSize = inputs.length;
        int[] predictions = new int[testSize];
        int chunkSize = (testSize + numThreads - 1) / numThreads;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch latch = new CountDownLatch(numThreads);
        
        try {
            for (int t = 0; t < numThreads; t++) {
                final int start = t * chunkSize;
                final int end = Math.min(start + chunkSize, testSize);
                
                executor.submit(() -> {
                    try {
                        for (int i = start; i < end; i++) {
                            synchronized (cnn) {
                                predictions[i] = cnn.predict(inputs[i]);
                            }
                        }
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Prediction interrupted", e);
        } finally {
            executor.shutdown();
        }
        
        return predictions;
    }
    
    /**
     * Benchmark comparison between sequential and parallel prediction
     * 
     * @param cnn The CNN model
     * @param inputs Test inputs
     * @return Benchmark results as formatted string
     */
    public static String benchmarkPrediction(CNN cnn, Tensor3D[] inputs) {
        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(50)).append("\n");
        sb.append("Prediction Benchmark (").append(inputs.length).append(" samples)\n");
        sb.append("=".repeat(50)).append("\n");
        
        // Sequential
        long startSeq = System.currentTimeMillis();
        int[] seqPredictions = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            seqPredictions[i] = cnn.predict(inputs[i]);
        }
        long seqTime = System.currentTimeMillis() - startSeq;
        sb.append(String.format("Sequential: %d ms (%.2f samples/sec)%n", 
            seqTime, inputs.length * 1000.0 / seqTime));
        
        // Parallel
        long startPar = System.currentTimeMillis();
        int[] parPredictions = parallelPredict(cnn, inputs);
        long parTime = System.currentTimeMillis() - startPar;
        sb.append(String.format("Parallel:   %d ms (%.2f samples/sec)%n", 
            parTime, inputs.length * 1000.0 / parTime));
        
        // Speedup
        double speedup = (double) seqTime / parTime;
        sb.append(String.format("Speedup:    %.2fx%n", speedup));
        
        // Verify correctness
        boolean match = true;
        for (int i = 0; i < inputs.length; i++) {
            if (seqPredictions[i] != parPredictions[i]) {
                match = false;
                break;
            }
        }
        sb.append("Results match: ").append(match ? "Yes" : "No").append("\n");
        sb.append("=".repeat(50));
        
        return sb.toString();
    }
    
    /**
     * Get recommended parallelism level based on dataset size and hardware
     * 
     * @param datasetSize Number of samples
     * @return Recommended number of threads
     */
    public static int getRecommendedParallelism(int datasetSize) {
        int processors = Runtime.getRuntime().availableProcessors();
        
        // Don't use more threads than we have samples
        // Also cap at processor count for CPU-bound operations
        return Math.min(processors, Math.max(1, datasetSize / 100));
    }
}
