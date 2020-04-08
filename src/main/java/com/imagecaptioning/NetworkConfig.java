/**
 * 
 */
package com.imagecaptioning;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Arnaud
 *
 */

public class NetworkConfig {

	private Params params = new Params();
	
	public ComputationGraph getNetworkConfig() {
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(params.getSeed())
				.optimizationAlgo(params.getOptimizationAlgorithm())
				.updater(params.getUpdater())
				.cacheMode(params.getCacheMode())
                .trainingWorkspaceMode(params.getWorkspaceMode())
                .inferenceWorkspaceMode(params.getWorkspaceMode())
				
				.graphBuilder()
				.allowDisconnected(true) // Need to figure out how not to use this without compromising the network architecture.
				
				//The 2 inputs of the network )
                .addInputs("imageIn", "descriptionOut")
                .setInputTypes(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels()), InputType.recurrent(params.getSequenceVectorSize()))
				
                // CNN block
                
				// block 1
                .addLayer("conv1_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nIn(params.getChannels())
                		.nOut(64)
                		.activation(params.getActivation())
                		.build(), "imageIn") //Output: (224-3+2)/1+1 = 224 -> 224*224*64
                .addLayer("conv1_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(64)
                		.activation(params.getActivation())
                		.build(), "conv1_1")  //Output: (224-3+2)/1+1 = 224 -> 224*224*64
                .addLayer("maxpool1_1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(2, 2)
                		.stride(2, 2)
                		.build(), "conv1_2")  //Output: (224-2+0)/2+1 = 112 -> 112*112*64
                
                // block 2
                .addLayer("conv2_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(128)
                		.activation(params.getActivation())
                		.build(), "maxpool1_1") //Output: (112-3+2)/1+1 = 112 -> 112*112*128
                .addLayer("conv2_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(128)
                		.activation(params.getActivation())
                		.build(), "conv2_1")   //Output: (112-3+2)/1+1 = 112 -> 112*112*128
                .addLayer("maxpool2_1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(2, 2)
                		.stride(2, 2)
                		.build(), "conv2_2")   //Output: (112-2+0)/2+1 = 56 -> -> 56*56*128
                
                 // block 3
                .addLayer("conv3_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(256)
                		.activation(params.getActivation())
                		.build(), "maxpool2_1")   //Output: (56-3+2)/1+1 = 56 -> 56*56*256
                .addLayer("conv3_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(256)
                		.activation(params.getActivation())
                		.build(), "conv3_1")    //Output: (56-3+2)/1+1 = 56 -> 56*56*256
                .addLayer("conv3_3", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(256)
                		.activation(params.getActivation())
                		.build(), "conv3_2")    //Output: (56-3+2)/1+1 = 56 -> 56*56*256
                .addLayer("maxpool3_1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(2, 2)
                		.stride(2, 2)
                		.build(), "conv3_3")    //Output: (56-2+0)/2+1 = 28 -> 28*28*256
                
                // block 4
                .addLayer("conv4_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "maxpool3_1")    //Output: (28-3+2)/1+1 = 28 -> 28*28*512
                .addLayer("conv4_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "conv4_1")    //Output: (28-3+2)/1+1 = 28 -> 28*28*512
                .addLayer("conv4_3", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "conv4_2")     //Output: (28-3+2)/1+1 = 28 -> 28*28*512
                .addLayer("maxpool4_1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(2, 2)
                		.stride(2, 2)
                		.build(), "conv4_3")     //Output: (28-2+0)/2+1 = 14 -> 14*14*512
                
                // block 5
                .addLayer("conv5_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "maxpool4_1")   //Output: (14-3+2)/1+1 = 14 -> 14*14*512
                .addLayer("conv5_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "conv5_1")     //Output: (14-3+2)/1+1 = 14 -> 14*14*512
                .addLayer("conv5_3", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(512)
                		.activation(params.getActivation())
                		.build(), "conv5_2")     //Output: (14-3+2)/1+1 = 14 -> 14*14*512
                .addLayer("maxpool5_1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(2, 2)
                		.stride(2, 2)
                		.build(), "conv5_3")  //Output: (14-2+0)/2+1 = 7 -> 7*7*512
                
                // block 6
                .addLayer("fc6_1", new DenseLayer.Builder() 
                		.nOut(4096)
                		.activation(params.getActivation())
                		.dropOut(0.5)
                		.build(), "maxpool5_1")
                .addLayer("fc6_2", new DenseLayer.Builder()
                		.nOut(4096)
                		.activation(params.getActivation())
                		.dropOut(0.5)
                		.build(), "fc6_1")
                
                
                // RNN block
                
                //All the inputs to the encoder will have size = batchSize * FeatureVectorSize * timesteps 
                .addLayer("encoder", new LSTM.Builder()
                		.nIn(4096)
                		.nOut(params.getNumHiddenNodes())
                		.activation(Activation.SOFTSIGN)
                		.build(), "fc6_2")  
                
                //Wrap the encoder LSTM layer with the LastTimeStep layer
                .addLayer("lastTimeStep", new LastTimeStep(new LSTM.Builder()
                		//.nIn(4096)  // not sure if these nIn and nOut are right
                		.nOut(params.getNumHiddenNodes())
                		.build()), "encoder")  // "fc6_2"
                
                 //Create a vertex that allows the duplication of 2d input to a 3d input
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("descriptionOut"), "lastTimeStep")
                
                //The inputs to the decoder will have size = size of output of last timestep of encoder (numHiddenNodes) + size of the other input
                .addLayer("decoder", new LSTM.Builder()
                		.nIn(params.getSequenceVectorSize() + params.getNumHiddenNodes())
                		.nOut(params.getNumHiddenNodes())
                		.activation(Activation.SOFTSIGN)
                		.build(), "descriptionOut", "duplicateTimeStep")
                
                .addLayer("output", new RnnOutputLayer.Builder()
                		.nIn(params.getNumHiddenNodes())
                		.nOut(params.getSequenceVectorSize())
                		.activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT)
                		.build(), "decoder")                
				
                
                //.inputPreProcessor("conv1_1", new RnnToCnnPreProcessor(params.getHeight(), params.getWidth(), params.getChannels()))
                //.inputPreProcessor("fc6_1", new CnnToFeedForwardPreProcessor(7, 7, 512))
                //.inputPreProcessor("encoder", new FeedForwardToRnnPreProcessor())
                
                .setOutputs("output")
				.build();
				
		return new ComputationGraph(conf);		
	}
}
