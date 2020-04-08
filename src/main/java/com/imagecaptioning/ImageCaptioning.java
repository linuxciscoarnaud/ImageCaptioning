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
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class ImageCaptioning {

	private static final Logger log = LoggerFactory.getLogger(ImageCaptioning.class);
	
	Params params = new Params();
  	protected static boolean save = true;
	
	public void execute(String[] args) throws Exception {
		
		// Loading the data		
  	    log.info("Loading data....");
  	    
		FileSplit trainFileSplit = new FileSplit(new File(params.getTrainImageDataDir()), NativeImageLoader.ALLOWED_FORMATS, params.getRng());
		int totalBatches = toIntExact(trainFileSplit.length()) / params.getBatchSize();
		List<Item> trainLabelList = readCsvDataFile(params.getTrainLabelsFile());
		CustomSequenceIterator trainDataIter = new CustomSequenceIterator(totalBatches, trainFileSplit, trainLabelList);
		
		// Building model...
		log.info("Building the model....");
		ComputationGraph network;
		String modelFilename = params.getBaseDataDir() + "/ImageCaptioning_Model.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading an existing trained model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
		} else {
			network = (new NetworkConfig().getNetworkConfig());
			network.init();
			
			//log.info(network.summary(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels())), InputType.recurrent(params.getFeatureVectorSize()));
			
			// Enabling the UI...
			// Initialize the user interface backend.
			//UIServer uiServer = UIServer.getInstance();
			// Configure where the information is to be stored. Here, we store in the memory.
			// It can also be saved in file using: StatsStorage statsStorage = new FileStatsStorage(File), if we intend to load and use it later.
			//StatsStorage statsStorage = new InMemoryStatsStorage();
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
            	trainDataIter.reset();
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
	
	/**
     *   Helper method that takes in a string representing the full path of the csv file containing the train/test data
     *   @param filePath.
     */
    private List<Item> readCsvDataFile(String filePath) {
    	
    	// Create a new list of line instances to be fill
		List<Item> itemList = new ArrayList<Item>();
    	BufferedReader fileReader = null;
    	
    	try {   		
    		String line = "";
    		
    		// Create the fileReader
    		fileReader = new BufferedReader(new FileReader(filePath));
    		
    		// Read the CSV file header to skip it
    		//fileReader.readLine();
    		
    		// Now read the file line by line starting from the first line
    		while ((line = fileReader.readLine()) != null) {
    			// Get all tokens available in line
				String[] tokens = line.split(",");
				if (tokens.length > 0) {
					Item item = new Item(tokens[0], tokens[1]);  // 0 for the image FileName index and 1 for the image Label index
					itemList.add(item);
				}
    		}
    	} catch (Exception e) {
    		log.info("!!! Data file not found !!!");
    		e.printStackTrace();
    	} finally {
    		try {
    			fileReader.close();
    		} catch (IOException ioe) {
    			log.info("!!! Error while closing fileReader !!!");
				ioe.printStackTrace();
    		}
    	}
    	
    	return itemList;
    }
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		new ImageCaptioning().execute(args);
	}
}
