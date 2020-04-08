/**
 * 
 */
package com.imagecaptioning;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;

/**
 * @author Arnaud
 *
 */

public class Params {

	// Parameters for network configuration
	private OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
	private IUpdater updater = new Nesterovs();
	private Activation activation = Activation.RELU;
	
	private CacheMode cacheMode = CacheMode.NONE;
	private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
	
	// Parameters for the training phase (hyper parameters)
	private int batchSize = 2;  
	private int nEpochs = 2;
	
	// Parameters for input data
	private int height = 224;
	private int width = 224;
	private int channels = 3; 
	
	private long seed = 123; // Integer for reproducibility of a random number generator
	private Random rng = new Random(seed);
	
	private int sequenceVectorSize = 91;
	private int maxCharForOutputs = 100;
	private int numHiddenNodes = 256; 
	
	// Data directory.
    private String baseDataDir = System.getProperty("user.dir")+ "/src/main/resources/data/";
    private String trainImageDataDir = System.getProperty("user.dir")+ "/src/main/resources/data/train/features/";
    private String trainLabelsFile = System.getProperty("user.dir")+ "/src/main/resources/data/train/labels/trainLabels.csv";
    private String testImageDataDir = System.getProperty("user.dir")+ "/src/main/resources/data/test/features/";
    private String testLabelsFile = System.getProperty("user.dir")+ "/src/main/resources/data/test/labels/testLabels.csv";
	
	// Getters
	
	public OptimizationAlgorithm getOptimizationAlgorithm() {
		return optimizationAlgorithm;
	}
	
	public IUpdater getUpdater() {
		return updater;
	}
	
	public Activation getActivation() {
		return activation;
	}
	
	public CacheMode getCacheMode() {
	 	return cacheMode;
	}
	 	
	public WorkspaceMode getWorkspaceMode() {
	 	return workspaceMode;
	}
	
	public int getBatchSize() {
		return batchSize;
	}
	
	public int getNEpochs() {
		return nEpochs;
	}
	
	public int getHeight() {
 		return height;
 	}
 	
 	public int getWidth( ) {
 		return width;
 	}
 	
 	public int getChannels() {
 		return channels;
 	}
	
	public long getSeed() {
		return seed;
	}
	
	public Random getRng() {
		return rng;
	}
	
	public int getSequenceVectorSize() {
		return sequenceVectorSize;
	}
	
	public int getMaxCharForOutputs() {
		return maxCharForOutputs;
	}
	
	public int getNumHiddenNodes() {
		return numHiddenNodes;
	}
	
	public String getBaseDataDir() {
		return baseDataDir ;
	}
	
	public String getTrainImageDataDir() {
		return trainImageDataDir;
	}
	
	public String getTrainLabelsFile() {
		return trainLabelsFile;
	}
	
	public String getImageTestDataDir() {
		return testImageDataDir;
	}
	
	public String getTestLabelsFile() {
		return testLabelsFile;
	}
}
