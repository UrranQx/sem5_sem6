package cnn.layers;

import cnn.utils.VectorOps;
import java.util.Random;

/**
 * Dense (Fully Connected) Layer
 * Implements matrix multiplication with biases
 */
public class DenseLayer implements Layer {
    private float[][] weights;  // [outputSize][inputSize]
    private float[] biases;
    private int inputSize;
    private int outputSize;
    
    private float[] lastInput;
    private float[] lastOutput;
    
    public DenseLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        initializeWeights();
    }
    
    private void initializeWeights() {
        Random rand = new Random(42);
        // Xavier/Glorot initialization
        float scale = (float) Math.sqrt(2.0 / (inputSize + outputSize));
        
        weights = new float[outputSize][inputSize];
        biases = new float[outputSize];
        
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = (float) (rand.nextGaussian() * scale);
            }
            biases[i] = 0.01f;
        }
    }
    
    @Override
    public Object forward(Object input) {
        float[] inputArray = (float[]) input;
        lastInput = inputArray;
        
        float[] output = VectorOps.matVecMul(weights, inputArray);
        output = VectorOps.add(output, biases);
        
        lastOutput = output;
        return output;
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        float[] gradOutput = (float[]) gradient;
        
        // Gradient for input
        float[][] weightsT = VectorOps.transpose(weights);
        float[] gradInput = VectorOps.matVecMul(weightsT, gradOutput);
        
        // Update weights and biases
        for (int i = 0; i < outputSize; i++) {
            biases[i] -= learningRate * gradOutput[i];
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] -= learningRate * gradOutput[i] * lastInput[j];
            }
        }
        
        return gradInput;
    }
    
    @Override
    public String getOutputShape() {
        return "Dense(" + outputSize + ")";
    }
}
