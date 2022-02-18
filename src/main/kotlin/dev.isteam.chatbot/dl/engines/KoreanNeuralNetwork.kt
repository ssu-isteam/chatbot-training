package dev.isteam.chatbot.dl.engines

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.math.pow

object KoreanNeuralNetwork {
    const val LSTM_LAYERSIZE = 200
    const val TBTT_SIZE = 50

    fun buildLogisticRegression(inputSize: Int, outputSize: Int): MultiLayerNetwork {
        var conf = NeuralNetConfiguration.Builder()
            .seed(1234)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Nesterovs(0.1, 0.9))
            .l2(0.0001)
            .list()
            .layer(
                0, OutputLayer.Builder()
                    .nIn(inputSize)
                    .nOut(outputSize)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .build()

        return MultiLayerNetwork(conf)
    }

    fun buildAutoEncoder_(inputSize: Int): MultiLayerNetwork {
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(1337)
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(0.05))
            .list()
            .build()
        return MultiLayerNetwork(modelConf)
    }

    fun buildAutoEncoder(inputSize: Int): MultiLayerNetwork {

        var modelConf = NeuralNetConfiguration.Builder()
            .seed(1337)
            .updater(Adam(1e-3))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, DenseLayer.Builder().activation(Activation.TANH).nIn(inputSize).nOut(1000).build())
            .layer(1, DenseLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(500).build())
            .layer(2, DenseLayer.Builder().activation(Activation.TANH).nIn(500).nOut(100).build())
            .layer(3, DenseLayer.Builder().activation(Activation.TANH).nIn(100).nOut(500).build())
            .layer(4, DenseLayer.Builder().activation(Activation.TANH).nIn(500).nOut(1000).build())
            .layer(
                5,
                OutputLayer.Builder().activation(Activation.TANH).nIn(1000).nOut(inputSize)
                    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build()
            )
            .build()
        return MultiLayerNetwork(modelConf)
    }

    fun buildDeepAutoEncoder(inputSize: Int): MultiLayerNetwork {

        val unit = 7
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(12356)
            .updater(Adam(1e-3))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
        var layerSizes = arrayListOf<Int>().also {
            var i = 0
            while (inputSize / unit.toDouble().pow(i).toInt() > 50) {
                it.add(inputSize / unit.toDouble().pow(i).toInt())
                i++
            }
        }
        (1 until layerSizes.size).forEach { index ->
            modelConf.layer(
                DenseLayer.Builder().nIn(layerSizes[index - 1]).nOut(layerSizes[index]).activation(Activation.TANH)
                    .build()
            )
            println("SHAPE: (${layerSizes[index - 1]},${layerSizes[index]})")
        }
        layerSizes.reverse()
        (1 until layerSizes.size).forEach { index ->
            modelConf.layer(
                DenseLayer.Builder().nIn(layerSizes[index - 1]).nOut(layerSizes[index]).activation(Activation.TANH)
                    .build()
            )
            println("SHAPE: (${layerSizes[index - 1]},${layerSizes[index]})")
        }
        modelConf.layer(
            OutputLayer.Builder().nIn(layerSizes.last()).nOut(layerSizes.last()).activation(Activation.RELU)
                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build()
        )

        return MultiLayerNetwork(modelConf.build())
    }

    fun buildVariationalAutoEncoder(inputSize: Int): MultiLayerNetwork {
        val unit = 7
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(12356)
            .updater(RmsProp(1e-2))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
        var layerSizes = arrayListOf<Int>().also {
            var i = 0
            while (inputSize / unit.toDouble().pow(i).toInt() > 50) {
                it.add(inputSize / unit.toDouble().pow(i).toInt())
                i++
            }
        }
        var vae = VariationalAutoencoder.Builder()
            .nIn(layerSizes.first()).nOut(layerSizes.last())
            .encoderLayerSizes(*layerSizes.slice(1 until layerSizes.size).toIntArray())
        layerSizes.reverse()
        vae.nOut(layerSizes.last()).decoderLayerSizes(*layerSizes.slice(1 until layerSizes.size).toIntArray())
            .lossFunction(Activation.TANH, LossFunctions.LossFunction.SQUARED_LOSS)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        modelConf.layer(vae.build())
        modelConf.layer(
            OutputLayer.Builder().nIn(layerSizes.last()).nOut(layerSizes.last()).activation(Activation.TANH)
                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build()
        )

        return MultiLayerNetwork(modelConf.build())
    }
    fun buildVariationalAutoEncoder_(inputSize: Int): MultiLayerNetwork {
        var modelConf = NeuralNetConfiguration.Builder()
            .seed(12356)
            .updater(RmsProp(1e-2))
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
        var layerSizes = intArrayOf( 1024, 512, 256, 128)
        var vae = VariationalAutoencoder.Builder()
            .nIn(inputSize).nOut(layerSizes.last())
            .encoderLayerSizes(*layerSizes)
        vae.nOut(inputSize).decoderLayerSizes(*layerSizes.reversedArray())
            .lossFunction(Activation.TANH, LossFunctions.LossFunction.SQUARED_LOSS)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        modelConf.layer(vae.build())
        modelConf.layer(
            OutputLayer.Builder().nIn(inputSize).nOut(inputSize).activation(Activation.TANH)
                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).build()
        )

        return MultiLayerNetwork(modelConf.build())
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
            .updater(Nesterovs(0.1, 0.9))
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
