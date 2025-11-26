package cnn.layers;

import cnn.utils.Tensor3D;
import cnn.utils.VectorOps;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * ReLU Activation Layer with Java Vector API optimization
 * Uses SIMD vectorization for accelerated computation
 */
public class ReLULayer implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private Tensor3D lastInput;
    private float[] lastInputFlat;
    private boolean is3D;
    
    @Override
    public Object forward(Object input) {
        if (input instanceof Tensor3D) {
            is3D = true;
            Tensor3D inputTensor = (Tensor3D) input;
            lastInput = inputTensor;
            return vectorizedRelu3D(inputTensor);
        } else {
            is3D = false;
            float[] inputArray = (float[]) input;
            lastInputFlat = inputArray;
            return VectorOps.relu(inputArray);
        }
    }
    
    /**
     * Vectorized ReLU for 3D tensor
     * Uses SIMD to process multiple elements in parallel
     */
    private Tensor3D vectorizedRelu3D(Tensor3D input) {
        Tensor3D result = new Tensor3D(input.channels, input.height, input.width);
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (int c = 0; c < input.channels; c++) {
            for (int h = 0; h < input.height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(input.width);
                
                // Vectorized processing
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, input.data[c][h], w);
                    v.max(zero).intoArray(result.data[c][h], w);
                }
                
                // Handle remaining elements
                for (; w < input.width; w++) {
                    result.data[c][h][w] = Math.max(0, input.data[c][h][w]);
                }
            }
        }
        
        return result;
    }
    
    @Override
    public Object backward(Object gradient, float learningRate) {
        if (is3D) {
            Tensor3D gradOutput = (Tensor3D) gradient;
            return vectorizedReluBackward3D(lastInput, gradOutput);
        } else {
            float[] gradOutput = (float[]) gradient;
            return VectorOps.reluBackward(lastInputFlat, gradOutput);
        }
    }
    
    /**
     * Vectorized ReLU backward pass for 3D tensor
     * Computes: gradOutput * (input > 0 ? 1 : 0)
     */
    private Tensor3D vectorizedReluBackward3D(Tensor3D input, Tensor3D gradOutput) {
        Tensor3D gradInput = new Tensor3D(input.channels, input.height, input.width);
        FloatVector zero = FloatVector.zero(SPECIES);
        
        for (int c = 0; c < input.channels; c++) {
            for (int h = 0; h < input.height; h++) {
                int w = 0;
                int upperBound = SPECIES.loopBound(input.width);
                
                // Vectorized processing
                for (; w < upperBound; w += SPECIES.length()) {
                    FloatVector vi = FloatVector.fromArray(SPECIES, input.data[c][h], w);
                    FloatVector vg = FloatVector.fromArray(SPECIES, gradOutput.data[c][h], w);
                    
                    // Create mask where input > 0
                    VectorMask<Float> mask = vi.compare(VectorOperators.GT, zero);
                    // Select gradient where mask is true, 0 otherwise
                    zero.blend(vg, mask).intoArray(gradInput.data[c][h], w);
                }
                
                // Handle remaining elements
                for (; w < input.width; w++) {
                    gradInput.data[c][h][w] = input.data[c][h][w] > 0 ? 
                        gradOutput.data[c][h][w] : 0;
                }
            }
        }
        
        return gradInput;
    }
    
    @Override
    public String getOutputShape() {
        return "ReLU";
    }
}
