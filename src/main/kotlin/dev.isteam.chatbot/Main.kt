package dev.isteam.chatbot

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.iterator.CorpusIterator
import dev.isteam.chatbot.dl.api.dataset.iterator.RawDataSetIterator
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File

val logger: Logger = LoggerFactory.getLogger("LSTM")

private const val TBPTT_SIZE = 25
private const val LEARNING_RATE = 1e-1
private const val RMS_DECAY = 0.95
private const val HIDDEN_LAYER_WIDTH = 512

private const val EMBEDDING_WIDTH = 128
private const val BATCH_SIZE = 32
private const val MACRO_BATCH_SIZE = 20
private const val MAX_LEN = 40
private const val EPOCH = 20
private const val MODEL_PATH = "model.h5"



private lateinit var tokenizerFactory: KoreanTokenizerFactory
private lateinit var vec: Word2Vec
private lateinit var corpus: PackedRawDataSet

private const val dataPath = "개인및관계.json"
fun main(args: Array<String>) {

    createDictionary(dataPath)

    val model = createGraph()
    train(model, corpus, tokenizerFactory,32)
}

fun createDictionary(path:String){
    val viveDataSetLoader = VIVEDataSetLoader(arrayOf(path))

    val packedRawDataSet = viveDataSetLoader.loadDialogues().get()[0]

    corpus = PackedRawDataSet(packedRawDataSet.dialogues.flatMap { it.rawDataSets }.toMutableList())



    tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    val rawDataSetIterator = RawDataSetIterator(corpus, RawDataSetIterator.IterativeType.QUESTION)

    if(! File("word2vec.bin").exists()){
        vec = Word2Vec.Builder()
            .minWordFrequency(5)
            .iterations(1)
            .epochs(1)
            .layerSize(50)
            .batchSize(10000)
            .workers(1000)
            .seed(42)
            .windowSize(5)
            .iterate(rawDataSetIterator)
            .tokenizerFactory(tokenizerFactory)
            .build()
        vec.fit()
        WordVectorSerializer.writeWord2VecModel(vec,"word2vec.bin")
    } else
        vec = WordVectorSerializer.readWord2VecModel(File("word2vec.bin"),true)


}
fun train(net:ComputationGraph, corpus:PackedRawDataSet, tokenizerFactory: KoreanTokenizerFactory, offset:Int){
    val uiServer = UIServer.getInstance()
    val storage = InMemoryStatsStorage()
    uiServer.attach(storage)

    net.setListeners(StatsListener(storage))

    val logsIterator = CorpusIterator(word2Vec = vec, corpus = corpus, tokenizerFactory = tokenizerFactory, batchSize = BATCH_SIZE, batchesPerMacroBatch = MACRO_BATCH_SIZE, maxLenPerSentence = MAX_LEN)
    for (epoch in 0 until EPOCH) {
        logger.info ("Epoch $epoch")
        if (epoch == 1) {
            logsIterator.setCurrentBatch(offset)
        } else {
            logsIterator.reset()
        }
        var lastPerc = 0
        while (logsIterator.hasNextMacroBatch()) {
            net.fit(logsIterator)
            logsIterator.nextMacroBatch()
            logger.info("Batch = " + logsIterator.batch())
            val newPerc: Int = logsIterator.batch() * 100 / logsIterator.totalBatches()
            if (newPerc != lastPerc) {
                logger.info("Epoch complete: $newPerc%")
                lastPerc = newPerc
            }
        }

    }
    net.save(File(MODEL_PATH),true)
}
fun createGraph() : ComputationGraph{
    val dictSize = vec.vocab.numWords()
    val updater = RmsProp(LEARNING_RATE)
    updater.rmsDecay = RMS_DECAY
    val builder = NeuralNetConfiguration.Builder()
    builder.updater(updater)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).miniBatch(true)
        .weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)

    val graphBuilder = builder.graphBuilder().backpropType(BackpropType.Standard)
        .tBPTTBackwardLength(TBPTT_SIZE).tBPTTForwardLength(TBPTT_SIZE)
    graphBuilder.addInputs("inputLine", "decoderInput")
        .setInputTypes(InputType.recurrent(dictSize.toLong()), InputType.recurrent(dictSize.toLong()))
        .addLayer(
            "embeddingEncoder",
            EmbeddingLayer.Builder().nIn(dictSize).nOut(EMBEDDING_WIDTH).build(),
            "inputLine"
        )
        .addLayer(
            "encoder",
            LSTM.Builder().nIn(EMBEDDING_WIDTH).nOut(HIDDEN_LAYER_WIDTH)
                .activation(Activation.TANH).build(),
            "embeddingEncoder"
        )
        .addVertex("thoughtVector", LastTimeStepVertex("inputLine"), "encoder")
        .addVertex("dup", DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
        .addVertex("merge", MergeVertex(), "decoderInput", "dup")
        .addLayer(
            "decoder",
            LSTM.Builder().nIn(dictSize + HIDDEN_LAYER_WIDTH)
                .nOut(HIDDEN_LAYER_WIDTH).activation(
                    Activation.TANH
                )
                .build(),
            "merge"
        )
        .addLayer(
            "output", RnnOutputLayer.Builder().nIn(HIDDEN_LAYER_WIDTH).nOut(dictSize).activation(
                Activation.SOFTMAX
            )
                .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder"
        )
        .setOutputs("output")

    val net = ComputationGraph(graphBuilder.build())
    net.init()
    return net
}