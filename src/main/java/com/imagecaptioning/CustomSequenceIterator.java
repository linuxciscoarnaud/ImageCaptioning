/**
 * 
 */
package com.imagecaptioning;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class CustomSequenceIterator implements MultiDataSetIterator {

	private static final Logger log = LoggerFactory.getLogger(CustomSequenceIterator.class);
    private MultiDataSetPreProcessor preProcessor;
    private final int batchSize;
    private final int totalBatches;
    private BaseImageLoader imageLoader;
    private List<Item> featuresLabelsMap = new ArrayList<Item>();
    private int offset = 0;
    
    private static Params params = new Params();
    
    private static int maxCharForOutputs = params.getMaxCharForOutputs();           
    public static int SEQ_VECTOR_DIM = params.getSequenceVectorSize();   
    public static final Map<String, Integer> oneHotMap = new HashMap<String, Integer>();
    public static final String[] oneHotOrder = new String[SEQ_VECTOR_DIM];
    
    private boolean toTestSet = false;
    private int currentBatch = 0;
    
    public CustomSequenceIterator(int totalBatches, FileSplit inputFileSplit, List<Item> labelList) {
    	
    	this.batchSize = params.getBatchSize();
        this.totalBatches = totalBatches;
        //this.imageLoader = new NativeImageLoader(params.getHeight(), params.getWidth(), params.getChannels(), imageTransform);
		this.imageLoader = new NativeImageLoader(params.getHeight(), params.getWidth(), params.getChannels());        
		this.featuresLabelsMap = featuresLabelsMap(inputFileSplit, labelList);
        oneHotEncoding();
    }
    
    @Override
    public MultiDataSet next(int sampleSize) {
    	INDArray encoderImage = null, decoderSeq = null, outputSeq = null;
    	int currentCount = 0;
    	List<INDArray> encoderImageList = new ArrayList<>();
        List<INDArray> decoderSeqList = new ArrayList<>();
        List<INDArray> outputSeqList = new ArrayList<>();
        Item item = new Item();
        
        while (currentCount < sampleSize) {
        	item = featuresLabelsMap.get(currentCount + offset);
        	
    		File image = new File(item.getPath());
    		String label = item.getImageLabel();
    		//System.out.print("Count: "+currentCount+ " ");
    		//System.out.println("File: "+image.toString());
    		try {
    			encoderImageList.add(imageLoader.asMatrix(image));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    		
    		String[] decoderInput = prepToString(label, true);
    		if (toTestSet) {
          		int k = 1;
          		while (k < decoderInput.length) {
          			decoderInput[k] = " ";
          			k++;
          		}
          	}
    		decoderSeqList.add(mapToOneHot(decoderInput));
    		
    		String[] decoderOutput = prepToString(label, false);
    		outputSeqList.add(mapToOneHot(decoderOutput));
    		
    		currentCount++;
        }
    	
        encoderImage = Nd4j.vstack(encoderImageList); 
        decoderSeq = Nd4j.vstack(decoderSeqList);
        outputSeq = Nd4j.vstack(outputSeqList);
        
        encoderImage = normalizeImages(encoderImage);
        
        INDArray[] inputs = new INDArray[]{encoderImage, decoderSeq};
        INDArray[] inputMasks = new INDArray[]{Nd4j.ones(sampleSize, 1, 1, 1), Nd4j.ones(sampleSize, maxCharForOutputs + 1 + 1)};
        INDArray[] labels = new INDArray[]{outputSeq};
        INDArray[] labelMasks = new INDArray[]{Nd4j.ones(sampleSize, maxCharForOutputs + 1 + 1)};
        currentBatch++;
        offset+=sampleSize;
        return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
    }
    
    @Override
    public void reset() {
        currentBatch = 0;
        toTestSet = false;
        offset = 0;
    }
    
    @Override
    public boolean resetSupported() {
        return true;
    }
    
    @Override
    public boolean asyncSupported() {
        return false;
    }
    
    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches;
    }
    
    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }
    
    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
    
    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }
    
    
    private List<Item> featuresLabelsMap(FileSplit inputFileSplit, List<Item> labelList) {
    	List<Item> flMap = new ArrayList<Item>();   	
    	Iterator<File> iter = new FileFromPathIterator(inputFileSplit.locationsPathIterator());
    	if (iter != null) {
    		while (iter.hasNext()) {
    			File image = iter.next();
    			String imageFileName = image.getName().substring(0, 13); // Assuming each image file name has 13 characters. Here, I am just removing the image file extension.
        		for (Item item : labelList) {
        			if (item.getImageFileName().equals(imageFileName)) {
        				item.setPath(image.toString()); // This is important
        				flMap.add(item);
        				break;
        			}
        		}
    		}
    	}
    	
    	return flMap;
    }
    
    
    /**
     * Normalizes image data. Scale them between 0 and 1
     * @param features
     */
    private INDArray normalizeImages(INDArray features) {
    	//double minRange = 0;
    	//double maxRange = 1;
        int maxBits = 8;
        
        double maxPixelVal = Math.pow(2, maxBits) - 1;
        features.divi(maxPixelVal);   //Scaled to 0->1
        
        //if (maxRange - minRange != 1) {
        	//features.muli(maxRange - minRange); //Scaled to minRange -> maxRange
        //}
        //if (minRange != 0) {
        	//features.addi(minRange); //Offset by minRange
        //}
        
        return features;
    }
    
    
    public String[] prepToString(String out, boolean goFirst) {
        int start, end;
        String[] decoded = new String[maxCharForOutputs + 1 + 1];
        if (goFirst) {
            decoded[0] = "GGGooo";
            start = 1;
            end = decoded.length - 1;
        } else {
            start = 0;
            end = decoded.length - 2;
            decoded[decoded.length - 1] = "EEEnnnddd";
        }

        int maxIndex = start;
        //add in digits
        for (int i = 0; i < out.length(); i++) {
            decoded[start + i] = Character.toString(out.charAt(i));
            maxIndex ++;
        }

        //needed padding
        while (maxIndex <= end) {
            decoded[maxIndex] = "_";
            maxIndex++;
        }
        return decoded;

    }
    
    
    /**
     *   This method takes in an array of strings and return a one hot encoded array of size 1 x FeatureVectorSize x timesteps
     *   Each element in the array indicates a time step
     *   @param toEncode the array of strings.
     */
    private static INDArray mapToOneHot(String[] toEncode) {

        INDArray ret = Nd4j.zeros(1, SEQ_VECTOR_DIM, toEncode.length);
        for (int i = 0; i < toEncode.length; i++) {
            ret.putScalar(0, oneHotMap.get(toEncode[i]), i, 1); 
        }

        return ret;
    }
    
    
    /**
     * One hot encoding map
    */
    private static void oneHotEncoding() {
    	
    	// Lower case letters of the alphabet
    	int k = 0;
    	for (char chUp = 'a'; chUp <= 'z'; chUp++) {
    		oneHotOrder[k] = Character.toString(chUp);
            oneHotMap.put(Character.toString(chUp), k);
            k++;
    	}
    	
    	// Capital letters of the alphabet
    	k = 26;
    	for (char chDown = 'A'; chDown <= 'Z'; chDown++) {
    		oneHotOrder[k] = Character.toString(chDown);
            oneHotMap.put(Character.toString(chDown), k);
            k++;
    	}
    	
    	// Digits
    	for (int i = 52; i <= 61; i++) {
            oneHotOrder[i] = String.valueOf(i-51);
            oneHotMap.put(String.valueOf(i-51), i);
        }
    	
    	// Special characters
    	oneHotOrder[62] = " ";
        oneHotMap.put(" ", 62);

        oneHotOrder[63] = "GGGooo";
        oneHotMap.put("GGGooo", 63);

        oneHotOrder[64] = "EEEnnnddd";
        oneHotMap.put("EEEnnnddd", 64);
        
        oneHotOrder[65] = ".";
        oneHotMap.put(".", 65);
        
        oneHotOrder[66] = "!";
        oneHotMap.put("!", 66);
        
        oneHotOrder[67] = "?";
        oneHotMap.put("?", 67);
        
        oneHotOrder[68] = "-";
        oneHotMap.put("-", 68);
        
        oneHotOrder[69] = "'";
        oneHotMap.put("'", 69);
        
        oneHotOrder[70] = "ô";
        oneHotMap.put("ô", 70);
        
        oneHotOrder[71] = "é";
        oneHotMap.put("é", 71);
        
        oneHotOrder[72] = "î";
        oneHotMap.put("î", 72);
        
        oneHotOrder[73] = "û";
        oneHotMap.put("û", 73);
        
        oneHotOrder[74] = "à";
        oneHotMap.put("à", 74);
        
        oneHotOrder[75] = "ç";
        oneHotMap.put("ç", 75);
        
        oneHotOrder[76] = "â";
        oneHotMap.put("â", 76);
        
        oneHotOrder[77] = "ê";
        oneHotMap.put("ê", 77);
        
        oneHotOrder[78] = "Ç";
        oneHotMap.put("Ç", 78);
        
        oneHotOrder[79] = "É";
        oneHotMap.put("É", 79);
        
        oneHotOrder[80] = "è";
        oneHotMap.put("è", 80);
        
        oneHotOrder[81] = "À";
        oneHotMap.put("À", 81);
        
        oneHotOrder[82] = "$";
        oneHotMap.put("$", 82);
        
        oneHotOrder[83] = "_";
        oneHotMap.put("_", 83);
        
        oneHotOrder[84] = ":";
        oneHotMap.put(":", 84);
        
        oneHotOrder[85] = "ù";
        oneHotMap.put("ù", 85);
        
        oneHotOrder[86] = "œ";
        oneHotMap.put("œ", 86);
        
        oneHotOrder[87] = "ï";
        oneHotMap.put("ï", 87);
        
        oneHotOrder[88] = "&";
        oneHotMap.put("&", 88);
        
        oneHotOrder[89] = "%";
        oneHotMap.put("%", 89);
        
        oneHotOrder[90] = "Ê";
        oneHotMap.put("Ê", 90);
    }
    
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }
}
