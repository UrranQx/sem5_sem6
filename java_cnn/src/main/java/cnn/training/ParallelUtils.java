package cnn.training;

import cnn.CNN;
import cnn.utils.Tensor3D;
import cnn.utils.ConfusionMatrix;

import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * Parallel utilities for CNN operations
 * 
 * Provides multi-threaded implementations for:
 * - Parallel prediction/inference
 * - Parallel evaluation with confusion matrix
 * - Batch prediction with thread pool
 * 
 * Note on thread safety:
 * CNN layers maintain internal state (lastInput, lastOutput) for backpropagation.
 * For inference-only operations, we use synchronization to ensure thread safety.
 * This serializes access to the CNN but still provides benefits:
 * - Better resource utilization (other threads can prepare data while one does inference)
 * - Simpler code without requiring layer cloning
 * 
 * For maximum parallelism in production, consider:
 * - Implementing thread-safe forward pass without state storage
 * - Creating separate CNN instances per thread (requires full clone support)
 * - Using batch inference with properly shaped tensors
 */
public class ParallelUtils {
    
    private static final int DEFAULT_PARALLELISM = Runtime.getRuntime().availableProcessors();
    
    /**
     * Parallel prediction using ForkJoinPool
     * 
     * Note: Due to CNN internal state requirements, predictions are synchronized.
     * The parallelism benefit comes from overlapping data preparation and I/O
     * operations with computation. For true parallel inference, use separate
     * CNN instances or implement stateless forward pass.
     * 
     * @param cnn The CNN model (access is synchronized for thread safety)
     * @param inputs Array of input tensors
     * @return Array of predicted class labels
     */
    public static int[] parallelPredict(CNN cnn, Tensor3D[] inputs) {
        int[] predictions = new int[inputs.length];
        
        // Use parallel streams - synchronization ensures thread safety
        // Trade-off: serialized CNN access, but parallel data handling
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
     * Sequential prediction for baseline comparison
     * 
     * @param cnn The CNN model
     * @param inputs Array of input tensors
     * @return Array of predicted class labels
     */
    public static int[] sequentialPredict(CNN cnn, Tensor3D[] inputs) {
        int[] predictions = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = cnn.predict(inputs[i]);
        }
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
        
        // Collect predictions (synchronized internally)
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
     * Provides finer control over threading but still requires synchronization
     * for CNN access.
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
                        // Each thread processes its chunk with synchronized CNN access
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
     * Note: Due to CNN synchronization requirements, parallel version may not
     * show significant speedup for pure inference. Main benefit is in mixed
     * workloads where data preparation can overlap with computation.
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
        int[] seqPredictions = sequentialPredict(cnn, inputs);
        long seqTime = System.currentTimeMillis() - startSeq;
        sb.append(String.format("Sequential: %d ms (%.2f samples/sec)%n", 
            seqTime, inputs.length * 1000.0 / Math.max(1, seqTime)));
        
        // Parallel (with synchronization)
        long startPar = System.currentTimeMillis();
        int[] parPredictions = parallelPredict(cnn, inputs);
        long parTime = System.currentTimeMillis() - startPar;
        sb.append(String.format("Parallel:   %d ms (%.2f samples/sec)%n", 
            parTime, inputs.length * 1000.0 / Math.max(1, parTime)));
        
        // Speedup
        double speedup = seqTime > 0 ? (double) seqTime / Math.max(1, parTime) : 1.0;
        sb.append(String.format("Speedup:    %.2fx%n", speedup));
        
        // Note about synchronization
        sb.append("Note: Parallelism limited by CNN synchronization\n");
        
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
