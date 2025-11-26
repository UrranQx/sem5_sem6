package cnn.layers;

import cnn.utils.VectorOps;

/**
 * Softmax Layer with Cross-Entropy Loss
 */
public class SoftmaxLayer implements Layer {
    private float[] lastOutput;
    
    @Override
    public Object forward(Object input) {
        float[] inputArray = (float[]) input;
        lastOutput = VectorOps.softmax(inputArray);
        return lastOutput;
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        // For softmax + cross-entropy, gradient is simply (output - target)
        // This is passed directly from loss calculation
        return gradient;
    }
    
    /**
     * Compute cross-entropy loss
     */
    public static float crossEntropyLoss(float[] predicted, float[] target) {
        float loss = 0;
        for (int i = 0; i < target.length; i++) {
            if (target[i] > 0) {
                loss -= target[i] * (float) Math.log(Math.max(predicted[i], 1e-10));
            }
        }
        return loss;
    }
    
    /**
     * Compute gradient of cross-entropy loss with softmax
     * Simplified: gradient = predicted - target
     */
    public static float[] crossEntropyGradient(float[] predicted, float[] target) {
        return VectorOps.subtract(predicted, target);
    }
    
    @Override
    public String getOutputShape() {
        return "Softmax";
    }
}
