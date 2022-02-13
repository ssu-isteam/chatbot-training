package dev.isteam.chatbot.dl.engines

import org.bytedeco.opencv.opencv_ml.LogisticRegression
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.learning.regularization.L2Regularization
import org.nd4j.linalg.learning.regularization.Regularization
import org.nd4j.linalg.lossfunctions.LossFunctions

object KoreanNeuralNetwork {
    const val LSTM_LAYERSIZE = 200
    const val TBTT_SIZE = 50

    fun buildLogisticRegression(inputSize: Int,outputSize: Int) : MultiLayerNetwork{
        var conf = NeuralNetConfiguration.Builder()
            .seed(1234)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Nesterovs(0.1,0.9))
            .l2(0.0001)
            .list()
            .layer(0, OutputLayer.Builder()
                .nIn(inputSize)
                .nOut(outputSize)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .build())
            .build()

        return MultiLayerNetwork(conf)
    }

    fun buildAutoEncoder_(inputSize: Int) : MultiLayerNetwork{
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(1337)
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(0.05))
            .regularization(listOf(L2Regularization(1e-4)))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0,AutoEncoder.Builder()
                .activation(Activation.SIGMOID)
                .nIn(inputSize).nOut(250).build())
            .layer(1,AutoEncoder.Builder()
                .activation(Activation.SIGMOID)
                .nIn(250).nOut(50).build())
            .layer(2,AutoEncoder.Builder()
                .activation(Activation.SIGMOID)
                .nIn(50).nOut(250).build())
            .layer(3,OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MSE)
                .nIn(250).nOut(inputSize)
                .build())
            .backpropType(BackpropType.Standard)
            .build()
        return MultiLayerNetwork(modelConf)
    }
    fun buildAutoEncoder(inputSize: Int) : MultiLayerNetwork{
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(1337)
            .updater(Adam(1e-4))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, DenseLayer.Builder().activation(Activation.TANH).nIn(inputSize).nOut(1000).build())
            .layer(1,   DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(500).build())
            .layer(2,  DenseLayer.Builder().activation(Activation.TANH).nIn(500).nOut(100).build())
            .layer(3,  DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(500).build())
            .layer(4,  DenseLayer.Builder().activation(Activation.TANH).nIn(500).nOut(1000).build())
            .layer(5,  OutputLayer.Builder().activation(Activation.RELU).nIn(1000).nOut(inputSize).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build())
            .build()
        return MultiLayerNetwork(modelConf)
    }
    fun buildNeuralNetworkLSTM(inputSize: Int, outputSize: Int): MultiLayerNetwork {
        var conf = NeuralNetConfiguration.Builder()
            .updater(Adam(0.1))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(13373)
            .dropOut(0.8)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(
                0, LSTM.Builder()

                    .nIn(inputSize).nOut(LSTM_LAYERSIZE)
                    .activation(Activation.TANH)
                    .build()
            )
            .layer(
                1, LSTM.Builder()
                    .nIn(LSTM_LAYERSIZE).nOut(LSTM_LAYERSIZE)
                    .activation(Activation.TANH)
                    .build()
            ).layer(
                2, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nIn(LSTM_LAYERSIZE).activation(Activation.SOFTMAX).nOut(outputSize).build()
            )
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(TBTT_SIZE).tBPTTBackwardLength(TBTT_SIZE)
            .build()
        return MultiLayerNetwork(conf)
    }

    fun buildNeuralNetwork(inputSize: Int, outputSize: Int): MultiLayerNetwork {
        var conf = NeuralNetConfiguration.Builder()
            .updater(Nesterovs(0.1,0.9))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(1337)
            .list()
            .layer(
                0, VariationalAutoencoder.Builder()
                    .nIn(inputSize)
                    .nOut(1024)
                    .encoderLayerSizes(1024, 512, 256, 128)
                    .decoderLayerSizes(128, 256, 512, 1024)
                    .lossFunction(Activation.RELU, LossFunctions.LossFunction.MSE)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .dropOut(0.8)
                    .build()
            )
            .layer(
                1, OutputLayer.Builder()
                    .nIn(1024).nOut(outputSize)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .build()
        return MultiLayerNetwork(conf)
    }
}