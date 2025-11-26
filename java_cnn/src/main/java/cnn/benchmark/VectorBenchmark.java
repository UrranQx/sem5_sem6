package cnn.benchmark;

import cnn.utils.VectorOps;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Random;

/**
 * Benchmark class to compare Java Vector API vs scalar implementations
 * Measures and compares execution time of various CNN operations
 */
public class VectorBenchmark {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    // Benchmark parameters
    private static final int WARMUP_ITERATIONS = 100;
    private static final int BENCHMARK_ITERATIONS = 1000;
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("Java Vector API vs Scalar Implementation Benchmark");
        System.out.println("=".repeat(70));
        System.out.println();
        System.out.println("Vector Species: " + SPECIES);
        System.out.println("Vector Length: " + SPECIES.length() + " floats");
        System.out.println("Warmup iterations: " + WARMUP_ITERATIONS);
        System.out.println("Benchmark iterations: " + BENCHMARK_ITERATIONS);
        System.out.println();
        
        // Run all benchmarks
        benchmarkArrayAddition();
        benchmarkArrayMultiplication();
        benchmarkDotProduct();
        benchmarkReLU();
        benchmarkSoftmax();
        benchmarkMatVecMul();
        benchmarkConvolution();
        benchmarkMaxPooling();
        benchmark3DReLU();
        
        System.out.println("=".repeat(70));
        System.out.println("Benchmark Complete");
        System.out.println("=".repeat(70));
    }
    
    /**
     * Benchmark array addition
     */
    private static void benchmarkArrayAddition() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Array Addition");
        System.out.println("-".repeat(70));
        
        int[] sizes = {256, 1024, 4096, 16384, 65536};
        
        for (int size : sizes) {
            Random rand = new Random(42);
            float[] a = randomArray(size, rand);
            float[] b = randomArray(size, rand);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.add(a, b);
                VectorOps.add(a, b);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.add(a, b);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.add(a, b);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("Addition", size, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark array multiplication
     */
    private static void benchmarkArrayMultiplication() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Array Multiplication");
        System.out.println("-".repeat(70));
        
        int[] sizes = {256, 1024, 4096, 16384, 65536};
        
        for (int size : sizes) {
            Random rand = new Random(42);
            float[] a = randomArray(size, rand);
            float[] b = randomArray(size, rand);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.multiply(a, b);
                VectorOps.multiply(a, b);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.multiply(a, b);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.multiply(a, b);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("Multiply", size, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark dot product
     */
    private static void benchmarkDotProduct() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Dot Product");
        System.out.println("-".repeat(70));
        
        int[] sizes = {256, 1024, 4096, 16384, 65536};
        
        for (int size : sizes) {
            Random rand = new Random(42);
            float[] a = randomArray(size, rand);
            float[] b = randomArray(size, rand);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.dot(a, b);
                VectorOps.dot(a, b);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.dot(a, b);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.dot(a, b);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("Dot", size, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark ReLU activation
     */
    private static void benchmarkReLU() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: ReLU Activation (1D)");
        System.out.println("-".repeat(70));
        
        int[] sizes = {256, 1024, 4096, 16384, 65536};
        
        for (int size : sizes) {
            Random rand = new Random(42);
            float[] a = randomArray(size, rand);
            // Make some values negative for ReLU
            for (int i = 0; i < size; i++) {
                a[i] -= 0.5f;
            }
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.relu(a);
                VectorOps.relu(a);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.relu(a);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.relu(a);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("ReLU", size, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark softmax
     */
    private static void benchmarkSoftmax() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Softmax");
        System.out.println("-".repeat(70));
        
        int[] sizes = {10, 100, 1000, 10000};
        
        for (int size : sizes) {
            Random rand = new Random(42);
            float[] a = randomArray(size, rand);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.softmax(a);
                VectorOps.softmax(a);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.softmax(a);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.softmax(a);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("Softmax", size, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark matrix-vector multiplication
     */
    private static void benchmarkMatVecMul() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Matrix-Vector Multiplication");
        System.out.println("-".repeat(70));
        
        int[][] sizes = {{128, 784}, {256, 512}, {512, 256}, {1024, 1024}};
        
        for (int[] size : sizes) {
            int rows = size[0];
            int cols = size[1];
            
            Random rand = new Random(42);
            float[][] matrix = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                matrix[i] = randomArray(cols, rand);
            }
            float[] vec = randomArray(cols, rand);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.matVecMul(matrix, vec);
                VectorOps.matVecMul(matrix, vec);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.matVecMul(matrix, vec);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.matVecMul(matrix, vec);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            printResult("MatVec " + rows + "x" + cols, rows * cols, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark convolution operation
     */
    private static void benchmarkConvolution() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Convolution");
        System.out.println("-".repeat(70));
        
        // Test configurations: [inputChannels, height, width, kernelSize]
        int[][] configs = {
            {1, 28, 28, 3},   // MNIST first conv
            {8, 14, 14, 3},   // MNIST second conv
            {16, 32, 32, 3},  // Larger input
            {32, 64, 64, 3}   // Even larger
        };
        
        for (int[] config : configs) {
            int inputChannels = config[0];
            int height = config[1];
            int width = config[2];
            int kernelSize = config[3];
            
            Random rand = new Random(42);
            
            // Create input with padding
            int paddedH = height + 2;
            int paddedW = width + 2;
            float[][][] padded = new float[inputChannels][paddedH][paddedW];
            for (int c = 0; c < inputChannels; c++) {
                for (int h = 0; h < paddedH; h++) {
                    padded[c][h] = randomArray(paddedW, rand);
                }
            }
            
            // Create kernel
            float[][][] kernel = new float[inputChannels][kernelSize][kernelSize];
            for (int c = 0; c < inputChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    kernel[c][kh] = randomArray(kernelSize, rand);
                }
            }
            
            float[][] outputScalar = new float[height][width];
            float[][] outputVector = new float[height][width];
            float bias = 0.1f;
            
            int iterations = Math.max(100, BENCHMARK_ITERATIONS / 10);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS / 10; i++) {
                ScalarOps.convolveFilter(padded, kernel, outputScalar, inputChannels, kernelSize, 1, bias);
                VectorOps.convolveFilter(padded, kernel, outputVector, inputChannels, kernelSize, 1, bias);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                ScalarOps.convolveFilter(padded, kernel, outputScalar, inputChannels, kernelSize, 1, bias);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                VectorOps.convolveFilter(padded, kernel, outputVector, inputChannels, kernelSize, 1, bias);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            String label = String.format("Conv %dx%dx%d k=%d", inputChannels, height, width, kernelSize);
            printResult(label, inputChannels * height * width, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark max pooling operation
     */
    private static void benchmarkMaxPooling() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: Max Pooling");
        System.out.println("-".repeat(70));
        
        int[][] configs = {
            {28, 28, 2},  // MNIST first pool
            {14, 14, 2},  // MNIST second pool
            {64, 64, 2},  // Larger input
            {32, 32, 4}   // Larger pool size
        };
        
        for (int[] config : configs) {
            int height = config[0];
            int width = config[1];
            int poolSize = config[2];
            
            Random rand = new Random(42);
            float[][] input = new float[height][width];
            for (int h = 0; h < height; h++) {
                input[h] = randomArray(width, rand);
            }
            
            int outH = (height - poolSize) / poolSize + 1;
            int outW = (width - poolSize) / poolSize + 1;
            
            float[][] outputScalar = new float[outH][outW];
            int[][] indicesHScalar = new int[outH][outW];
            int[][] indicesWScalar = new int[outH][outW];
            
            float[][] outputVector = new float[outH][outW];
            int[][] indicesHVector = new int[outH][outW];
            int[][] indicesWVector = new int[outH][outW];
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ScalarOps.maxPool2D(input, outputScalar, indicesHScalar, indicesWScalar, poolSize, poolSize);
                VectorOps.maxPool2D(input, outputVector, indicesHVector, indicesWVector, poolSize, poolSize);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                ScalarOps.maxPool2D(input, outputScalar, indicesHScalar, indicesWScalar, poolSize, poolSize);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                VectorOps.maxPool2D(input, outputVector, indicesHVector, indicesWVector, poolSize, poolSize);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            String label = String.format("Pool %dx%d p=%d", height, width, poolSize);
            printResult(label, height * width, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Benchmark 3D ReLU (for tensor operations)
     */
    private static void benchmark3DReLU() {
        System.out.println("-".repeat(70));
        System.out.println("Benchmark: 3D ReLU (Tensor)");
        System.out.println("-".repeat(70));
        
        int[][] configs = {
            {8, 28, 28},   // After first conv
            {16, 14, 14},  // After second conv
            {32, 32, 32},  // Larger
            {64, 64, 64}   // Even larger
        };
        
        for (int[] config : configs) {
            int channels = config[0];
            int height = config[1];
            int width = config[2];
            
            Random rand = new Random(42);
            float[][][] input = new float[channels][height][width];
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    input[c][h] = randomArray(width, rand);
                    // Make some negative
                    for (int w = 0; w < width; w++) {
                        input[c][h][w] -= 0.5f;
                    }
                }
            }
            
            float[][][] outputScalar = new float[channels][height][width];
            float[][][] outputVector = new float[channels][height][width];
            
            int iterations = Math.max(100, BENCHMARK_ITERATIONS / 5);
            
            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS / 10; i++) {
                ScalarOps.relu3D(input, outputScalar);
                VectorOps.relu3D(input, outputVector);
            }
            
            // Scalar benchmark
            long scalarStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                ScalarOps.relu3D(input, outputScalar);
            }
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Vector benchmark
            long vectorStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                VectorOps.relu3D(input, outputVector);
            }
            long vectorTime = System.nanoTime() - vectorStart;
            
            String label = String.format("ReLU3D %dx%dx%d", channels, height, width);
            printResult(label, channels * height * width, scalarTime, vectorTime);
        }
        System.out.println();
    }
    
    /**
     * Create random float array
     */
    private static float[] randomArray(int size, Random rand) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = rand.nextFloat();
        }
        return arr;
    }
    
    /**
     * Print benchmark result
     */
    private static void printResult(String operation, int size, long scalarNanos, long vectorNanos) {
        double scalarMs = scalarNanos / 1_000_000.0;
        double vectorMs = vectorNanos / 1_000_000.0;
        double speedup = (double) scalarNanos / vectorNanos;
        
        String speedupStr;
        if (speedup >= 1.0) {
            speedupStr = String.format("%.2fx faster", speedup);
        } else {
            speedupStr = String.format("%.2fx slower", 1.0 / speedup);
        }
        
        System.out.printf("%-25s | Size: %8d | Scalar: %8.2fms | Vector: %8.2fms | %s%n",
                operation, size, scalarMs, vectorMs, speedupStr);
    }
}
