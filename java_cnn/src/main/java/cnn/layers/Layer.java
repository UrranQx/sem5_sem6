package cnn.layers;

import cnn.utils.Tensor3D;

/**
 * Interface for CNN layers
 */
public interface Layer {
    /**
     * Forward pass
     */
    Object forward(Object input);
    
    /**
     * Backward pass - returns gradient for previous layer
     */
    Object backward(Object gradient, float learningRate);
    
    /**
     * Get output shape description
     */
    String getOutputShape();
}
