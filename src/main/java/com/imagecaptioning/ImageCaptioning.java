/**
 * 
 */
package com.imagecaptioning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.toIntExact;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class ImageCaptioning {

	private static final Logger log = LoggerFactory.getLogger(ImageCaptioning.class);
	
	private Params params = new Params();
	private Utils utils = new Utils();
  	protected static boolean save = true;
  	int totalBatches;
	
	public void execute(String[] args) throws Exception {
		
		// Loading the data		
  	    log.info("Loading data....");
  	    
		FileSplit trainFileSplit = new FileSplit(new File(params.getTrainImageDataDir()), NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		totalBatches = toIntExact(trainFileSplit.length()) / params.getBatchSize();
		List<Item> trainLabelList = utils.readCsvDataFile(params.getTrainLabelsFile());
		CustomSequenceIterator trainDataIter = new CustomSequenceIterator(totalBatches, trainFileSplit, trainLabelList);
		
		FileSplit testFileSplit = new FileSplit(new File(params.getTestImageDataDir()), NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		totalBatches = toIntExact(testFileSplit.length()) / params.getBatchSize();
		List<Item> testLabelList = utils.readCsvDataFile(params.getTestLabelsFile());
		CustomSequenceIterator testDataIter = new CustomSequenceIterator(totalBatches, testFileSplit, testLabelList);
		
		// Building model...
		log.info("Building the model....");
		ComputationGraph network;
		String modelFilename = params.getBaseDataDir() + "/ImageCaptioning_Model.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading an existing trained model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
			
			// Run test with the current model
        	//runTest(testDataIter, new Image2SeqPredicter(network), true);
		} else {
			network = (new NetworkConfig().getNetworkConfig());
			network.init();
			
			log.info(network.summary());
			
			// Enabling the UI...
			// Initialize the user interface backend.
			//UIServer uiServer = UIServer.getInstance();
			// Configure where the information is to be stored. Here, we store in the memory.
			// It can also be saved in file using: StatsStorage statsStorage = new FileStatsStorage(File), if we intend to load and use it later.
			//StatsStorage statsStorage = new InMemoryStatsStorage();
			//StatsStorage statsStorage = new FileStatsStorage(new File(params.getTrainInfosFile()));
			// Attach the StatsStorage instance to the UI. This allows the contents of the StatsStorage to be visualized.
			//uiServer.attach(statsStorage);
			// Add the StatsListener to collect information from the network, as it trains.
			//network.setListeners(new StatsListener(statsStorage));
			
			// Training...
        	log.info("Training the model....");
        	int iEpoch = 0;
        	while (iEpoch < params.getNEpochs()) {
        		log.info("\n");
        		network.fit(trainDataIter);
            	log.info("** EPOCH " + iEpoch + " COMPLETED **\n");
            	
            	// Run test with the current model
            	//runTest(testDataIter, new Image2SeqPredicter(network), true);
            	trainDataIter.reset();
            	log.info("\n");
                iEpoch++;
        	}
        	
        	// Save model...
            log.info("Saving the model....");
            if (save) {
	        	ModelSerializer.writeModel(network, modelFilename, true);
	        }
            log.info("Model has been saved....");
		}
		
	}
	
	
	private void runTest(CustomSequenceIterator tDataIter, Image2SeqPredicter predictor, boolean print) {
		//MultiDataSet testData = tDataIter.getTestData();
		//INDArray predictions = predictor.output(testData);
		//encode_decode_eval(predictions, testData.getFeatures()[0], testData.getLabels()[0]);
		
		if (print) {
			log.info("\n");
    		log.info("Printing stepping through the decoder for a minibatch of size 4 :");
        	MultiDataSet testDataForStepping = tDataIter.getTestData(4);
        	predictor.output(testDataForStepping, true);
		}
	}
	
	
	private void encode_decode_eval(INDArray predictions, INDArray questions, INDArray answers) {
		int nTests = (int)predictions.size(0);
		
		String[] questionS = CustomSequenceIterator.oneHotDecode(questions);
        String[] answersS = CustomSequenceIterator.oneHotDecode(answers);
        String[] predictionS = CustomSequenceIterator.oneHotDecode(predictions);
        
        for (int iTest = 0; iTest < nTests; iTest++) {
        	log.info(utils.cleanUp2(questionS[iTest]) + "  ---->  " + "' "+ 
                     utils.cleanUp1(predictionS[iTest]) + " '" + " COORECT ANSWER: " + 
        			 utils.cleanUp1(answersS[iTest]));
        }
        
	}
	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		new ImageCaptioning().execute(args);
	}
}
