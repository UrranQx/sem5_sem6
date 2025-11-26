package cnn;

import cnn.layers.*;
import cnn.utils.Tensor3D;
import java.util.ArrayList;
import java.util.List;

/**
 * Convolutional Neural Network
 * Combines layers and provides training/inference methods
 */
public class CNN {
    private List<Layer> layers;
    private float learningRate;
    
    public CNN(float learningRate) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
    }
    
    public void addLayer(Layer layer) {
        layers.add(layer);
    }
    
    /**
     * Forward pass through all layers
     */
    public float[] forward(Tensor3D input) {
        Object current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return (float[]) current;
    }
    
    /**
     * Backward pass through all layers
     */
    public void backward(float[] gradient) {
        Object currentGrad = gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentGrad = layers.get(i).backward(currentGrad, learningRate);
        }
    }
    
    /**
     * Train on a single sample
     */
    public float trainStep(Tensor3D input, float[] target) {
        // Forward pass
        float[] output = forward(input);
        
        // Compute loss
        float loss = SoftmaxLayer.crossEntropyLoss(output, target);
        
        // Compute gradient
        float[] gradient = SoftmaxLayer.crossEntropyGradient(output, target);
        
        // Backward pass
        backward(gradient);
        
        return loss;
    }
    
    /**
     * Predict class for input
     */
    public int predict(Tensor3D input) {
        float[] output = forward(input);
        return argmax(output);
    }
    
    /**
     * Get output probabilities
     */
    public float[] predictProba(Tensor3D input) {
        return forward(input);
    }
    
    /**
     * Find index of maximum value
     */
    public static int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Print model architecture
     */
    public void summary() {
        System.out.println("=" .repeat(50));
        System.out.println("CNN Architecture");
        System.out.println("=" .repeat(50));
        for (int i = 0; i < layers.size(); i++) {
            System.out.printf("Layer %d: %s%n", i + 1, layers.get(i).getOutputShape());
        }
        System.out.println("=" .repeat(50));
    }
    
    /**
     * Set learning rate
     */
    public void setLearningRate(float lr) {
        this.learningRate = lr;
    }
}
