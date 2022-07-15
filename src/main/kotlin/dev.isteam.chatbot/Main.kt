package dev.isteam.chatbot

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.iterator.CorpusIterator
import dev.isteam.chatbot.dl.api.dataset.iterator.RawDataSetIterator
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.preprocessor.MeanEmbeddingVectorizer
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.api.MaskState
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.RNNFormat
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.ui.VertxUIServer
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import java.lang.reflect.Field
import java.lang.reflect.Modifier

val logger: Logger = LoggerFactory.getLogger("LSTM")

/***
 * Word2Vec Settings
 */
private const val WORD2VEC_BATCH_SIZE = 5000
private const val WORD2VEC_EPOCH = 3
private const val WINDOW_SIZE = 10
private const val LAYER_SIZE = WINDOW_SIZE
private const val WORKERS = 100

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
private val word2VecFile = File("word2vec.bin")


private const val dataPath = "개인및관계.json"
fun main(args: Array<String>) {
    /***
     * Preparing dataset
     */

    val rawDataSets = VIVEDataSetLoader(arrayOf(dataPath)).loadDialogues().get().flatMap { it.dialogues }.flatMap { it.rawDataSets }.toMutableList()

    val rawDataSetIterator = RawDataSetIterator(PackedRawDataSet(rawDataSets = rawDataSets),RawDataSetIterator.IterativeType.QUESTION)
    /***
     * Preparing word2vec and tfid vectorizer
     */
    val tokenizerFactory = KoreanTokenizerFactory().apply {
        tokenPreProcessor = KoreanTokenPreprocessor()
    }


    val koreanTfidfVectorizer = KoreanTfidfVectorizer(packedRawDataSet = PackedRawDataSet(rawDataSets), koreanTokenizerFactory = tokenizerFactory)

    val word2Vec:Word2Vec

    if(word2VecFile.exists()){
        word2Vec = WordVectorSerializer.readWord2VecModel(word2VecFile,true)
    }else{
        word2Vec = Word2Vec.Builder()
            .allowParallelTokenization(true)
            .tokenizerFactory(tokenizerFactory)
            .iterate(rawDataSetIterator)
            .batchSize(WORD2VEC_BATCH_SIZE)
            .epochs(WORD2VEC_EPOCH)
            .windowSize(WINDOW_SIZE)
            .layerSize(LAYER_SIZE)
            .workers(WORKERS)
            .build()

        word2Vec.fit()
        word2Vec.buildVocab()
        WordVectorSerializer.writeWord2Vec(word2Vec, word2VecFile.outputStream())
    }

    /***
     * Preprocessing dataset
     */
    val srcCorpus = rawDataSets.filterIndexed { index, _ -> index % 2 == 0 }
    val tarCorpus = rawDataSets.filterIndexed { index, _ -> index % 2 != 0 }

    //These will be used in padding inputs.
    val srcMaxLen = srcCorpus.maxOf { tokenizerFactory.create(it.question).countTokens() }
    val tarMaxLen  = tarCorpus.maxOf { tokenizerFactory.create(it.question).countTokens() }


    /***
     * Preparing network
     */
    val modelConfig = NeuralNetConfiguration.Builder()
        .updater(AdaGrad(0.1))
        .graphBuilder()
        .addInputs("encoderInput","decoderInput")
        .setInputTypes(InputType.recurrent(WINDOW_SIZE.toLong(),srcMaxLen.toLong()), InputType.recurrent(tarMaxLen.toLong()))
        .addLayer("encoder",LSTM.Builder().nOut(tarMaxLen).build(),"encoderInput")
        .addLayer("decoder",LSTM.Builder().nOut(tarMaxLen) .build(),"decoderInput","encoder")
        .addLayer("outputLayer", RnnOutputLayer.Builder().nIn(tarMaxLen) .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS) .activation(Activation.TANH).nOut(tarMaxLen.toLong()).build(),"decoder")
        .inputPreProcessor("decoder",DecoderInputPreProcessor())
        .setOutputs("outputLayer")
        .build()

    val model = ComputationGraph(modelConfig)

    model.init()


    for(j in 0..EPOCH){
        for(i in 0..srcCorpus.size){
            val src = srcCorpus[i].question
            val tar = tarCorpus[i].question

            var srcVector = getSequenceVector(tokenizerFactory.create(src).tokens,word2Vec)
            srcVector = Nd4j.pad(srcVector,srcMaxLen)

            var tarVector = getSequenceVector(tokenizerFactory.create(tar).tokens,word2Vec)
            tarVector = Nd4j.pad(tarVector,tarMaxLen)

            val dataSet = MultiDataSet(arrayOf(srcVector,tarVector), arrayOf(tarVector))
            model.fit(dataSet)
            println(model.score())
        }
    }




}

fun getSequenceVector(tokens:List<String>, word2Vec: Word2Vec) : INDArray{
    val vector = Nd4j.create(tokens.size, LAYER_SIZE)

    tokens.forEachIndexed { index, s -> if(word2Vec.hasWord(s)){ vector.put(index,Nd4j.createFromArray(word2Vec.getWordVector(s).toTypedArray()))} }
    return vector
}

private class DecoderInputPreProcessor : InputPreProcessor{
    override fun clone(): InputPreProcessor {
        return DecoderInputPreProcessor()
    }

    /**
     * Pre preProcess input/activations for a multi layer network
     * @param input the input to pre preProcess
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the processed input. Note that the returned array should be placed in the
     * [org.deeplearning4j.nn.workspace.ArrayType.ACTIVATIONS] workspace via the workspace manager
     */
    override fun preProcess(input: INDArray?, miniBatchSize: Int, workspaceMgr: LayerWorkspaceMgr?): INDArray {
        println(input?.shapeInfoToString())
        return input!!
    }

    /**Reverse the preProcess during backprop. Process Gradient/epsilons before
     * passing them to the layer below.
     * @param output which is a pair of the gradient and epsilon
     * @param miniBatchSize Minibatch size
     * @param workspaceMgr Workspace manager
     * @return the reverse of the pre preProcess step (if any). Note that the returned array should be
     * placed in [org.deeplearning4j.nn.workspace.ArrayType.ACTIVATION_GRAD] workspace via the
     * workspace manager
     */
    override fun backprop(output: INDArray?, miniBatchSize: Int, workspaceMgr: LayerWorkspaceMgr?): INDArray {
        return output!!
    }

    /**
     * For a given type of input to this preprocessor, what is the type of the output?
     *
     * @param inputType    Type of input for the preprocessor
     * @return             Type of input after applying the preprocessor
     */
    override fun getOutputType(inputType: InputType?): InputType {
        return inputType!!
    }

    override fun feedForwardMaskArray(
        maskArray: INDArray?,
        currentMaskState: MaskState?,
        minibatchSize: Int
    ): Pair<INDArray, MaskState> {
        return Pair(maskArray,currentMaskState)
    }

}