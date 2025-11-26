package cnn.utils;

/**
 * Confusion Matrix for classification evaluation
 */
public class ConfusionMatrix {
    private int[][] matrix;
    private int numClasses;
    private String[] classNames;
    
    public ConfusionMatrix(int numClasses) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.classNames = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classNames[i] = String.valueOf(i);
        }
    }
    
    public ConfusionMatrix(int numClasses, String[] classNames) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.classNames = classNames;
    }
    
    /**
     * Add a prediction to the matrix
     */
    public void add(int actual, int predicted) {
        matrix[actual][predicted]++;
    }
    
    /**
     * Add batch of predictions
     */
    public void addBatch(int[] actuals, int[] predictions) {
        for (int i = 0; i < actuals.length; i++) {
            add(actuals[i], predictions[i]);
        }
    }
    
    /**
     * Get total correct predictions
     */
    public int getCorrect() {
        int correct = 0;
        for (int i = 0; i < numClasses; i++) {
            correct += matrix[i][i];
        }
        return correct;
    }
    
    /**
     * Get total predictions
     */
    public int getTotal() {
        int total = 0;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                total += matrix[i][j];
            }
        }
        return total;
    }
    
    /**
     * Get accuracy
     */
    public double getAccuracy() {
        return (double) getCorrect() / getTotal();
    }
    
    /**
     * Get precision for a class
     */
    public double getPrecision(int classIdx) {
        int tp = matrix[classIdx][classIdx];
        int fp = 0;
        for (int i = 0; i < numClasses; i++) {
            if (i != classIdx) {
                fp += matrix[i][classIdx];
            }
        }
        return tp + fp > 0 ? (double) tp / (tp + fp) : 0;
    }
    
    /**
     * Get recall for a class
     */
    public double getRecall(int classIdx) {
        int tp = matrix[classIdx][classIdx];
        int fn = 0;
        for (int j = 0; j < numClasses; j++) {
            if (j != classIdx) {
                fn += matrix[classIdx][j];
            }
        }
        return tp + fn > 0 ? (double) tp / (tp + fn) : 0;
    }
    
    /**
     * Get F1 score for a class
     */
    public double getF1Score(int classIdx) {
        double precision = getPrecision(classIdx);
        double recall = getRecall(classIdx);
        return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }
    
    /**
     * Print confusion matrix
     */
    public void print() {
        System.out.println("\n" + "=" .repeat(60));
        System.out.println("CONFUSION MATRIX");
        System.out.println("=" .repeat(60));
        
        // Header
        System.out.print("Actual\\Pred ");
        for (int j = 0; j < numClasses; j++) {
            System.out.printf("%6s", classNames[j]);
        }
        System.out.println();
        
        // Matrix rows
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%11s ", classNames[i]);
            for (int j = 0; j < numClasses; j++) {
                if (i == j) {
                    System.out.printf("[%4d]", matrix[i][j]);
                } else {
                    System.out.printf(" %4d ", matrix[i][j]);
                }
            }
            System.out.println();
        }
        
        System.out.println("-" .repeat(60));
        System.out.printf("Overall Accuracy: %.2f%% (%d/%d)%n", 
            getAccuracy() * 100, getCorrect(), getTotal());
        
        // Per-class metrics
        System.out.println("\nPer-class metrics:");
        System.out.printf("%-10s %10s %10s %10s%n", "Class", "Precision", "Recall", "F1-Score");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-10s %10.4f %10.4f %10.4f%n", 
                classNames[i], getPrecision(i), getRecall(i), getF1Score(i));
        }
        System.out.println("=" .repeat(60));
    }
}
