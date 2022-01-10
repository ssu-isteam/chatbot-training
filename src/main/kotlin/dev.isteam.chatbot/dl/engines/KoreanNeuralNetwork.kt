package dev.isteam.chatbot.dl.engines

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions

object KoreanNeuralNetwork {
    const val LSTM_LAYERSIZE = 200
    const val TBTT_SIZE = 50
    fun buildNeuralNetworkLSTM(inputSize: Int, outputSize: Int): MultiLayerNetwork {
        var conf = NeuralNetConfiguration.Builder()
            .updater(AdaGrad(0.01))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(13373)
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
            .updater(AdaGrad(0.01))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(1337)
            .weightInit(WeightInit.XAVIER)
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
                    .activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .build()
            )
            .backpropType(BackpropType.Standard)
            .build()
        return MultiLayerNetwork(conf)
    }
}