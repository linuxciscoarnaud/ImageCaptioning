/**
 * 
 */
package com.imagecaptioning;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */

public class Image2SeqPredicter {

	private static final Logger log = LoggerFactory.getLogger(Image2SeqPredicter.class);
    private ComputationGraph net;
    private INDArray decoderInputTemplate = null;
    
    public Image2SeqPredicter(ComputationGraph net) { 
        this.net = net;
    }
    
    
    public INDArray output(MultiDataSet testSet) {
    	
    	if (testSet.getFeatures()[0].size(0) > 2) {
            return output(testSet, false);
        } else {
            return output(testSet, true);
        }
    }
    
    
    public INDArray output(MultiDataSet testSet, boolean print) {
    	
    	INDArray correctOutput = testSet.getLabels()[0];
        INDArray ret = Nd4j.zeros(correctOutput.shape());
        decoderInputTemplate = testSet.getFeatures()[1].dup();
        
        int currentStepThrough = 0;
        int stepThroughs = (int)correctOutput.size(2)-1;
        
        while (currentStepThrough < stepThroughs) {       	
        	if (print) {
        		log.info("In time step " + currentStepThrough);
            	log.info("\tEncoder input and Decoder input:");
            	log.info(CustomSequenceIterator.mapToString(testSet.getFeatures()[0], decoderInputTemplate, " +  "));
        	}
        	ret = stepOnce(testSet, currentStepThrough);
        	if (print) {
        		log.info("\tDecoder output:");
            	log.info("\t"+String.join("\n\t",CustomSequenceIterator.oneHotDecode(ret)));
        	}
        	currentStepThrough++;
        }
        
        ret = net.output(false, testSet.getFeatures()[0], decoderInputTemplate)[0];
        if (print) {
        	log.info("Final time step " + currentStepThrough);
        	log.info("\tEncoder input and Decoder input:");
        	log.info(CustomSequenceIterator.mapToString(testSet.getFeatures()[0], decoderInputTemplate, " +  "));
        	log.info("\tDecoder output:");
        	log.info("\t"+String.join("\n\t",CustomSequenceIterator.oneHotDecode(ret)));
        }
        
        return ret;
    }
    
    
    private INDArray stepOnce(MultiDataSet testSet, int n) {
    	INDArray currentOutput = net.output(false, testSet.getFeatures()[0], decoderInputTemplate)[0];
        copyTimeSteps(n, currentOutput, decoderInputTemplate);
        
        return currentOutput;
    }
    
    
    private void copyTimeSteps(int t, INDArray fromArr, INDArray toArr) {
    	INDArray fromView = fromArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(0,t,true));
        INDArray toView = toArr.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(1,t+1,true));
        toView.assign(fromView.dup());
    }
}
