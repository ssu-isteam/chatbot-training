import dev.isteam.chatbot.dl.api.dataset.Word2VecRawDataSet
import dev.isteam.chatbot.dl.api.dataset.iterator.CharacterIterator
import dev.isteam.chatbot.dl.api.dataset.iterator.RawDataSetIterator
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.api.vector.Word2VecDataSource
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Paths
import java.security.SecureRandom
import kotlin.random.Random


val logger: Logger = LoggerFactory.getLogger("Main")

fun main2(args: Array<String>) {
    val motherPath = ""
    val files =
        arrayOf("개인및관계.json"/*"미용과건강.json","상거래(쇼핑).json","시사교육.json","식음료.json","여가생활.json","일과직업.json","주거와생활.json","행사.json"*/)

    logger.info("Reading files....")

    var viveDataSetLoader = VIVEDataSetLoader(files.map { Paths.get(motherPath, it).toString() }.toTypedArray())

    var packedRawDataSet = viveDataSetLoader.loadDialogues().get()[0]
    /*
    packedRawDataSet.dialogues = packedRawDataSet.dialogues.subList(0, 100)
    packedRawDataSet.rawDataSets = packedRawDataSet.dialogues.map { it.rawDataSets }.flatten().toMutableList()
    packedRawDataSet.rawDataSets = filter(packedRawDataSet,true)
     */
    logger.info("Reading files completed. Total count: ${packedRawDataSet.dialogues.size}")

    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var batchSize = 1000

    logger.info("Starting fitting tfidf vectorizer....")



    var koreanTfidfVectorizer =
        KoreanTfidfVectorizer(packedRawDataSet = packedRawDataSet, koreanTokenizerFactory = tokenizerFactory)
   koreanTfidfVectorizer.fit()


    var rawDataSetIterator = RawDataSetIterator(packedRawDataSet,RawDataSetIterator.IterativeType.QUESTION)


    val vec: Word2Vec = Word2Vec.Builder()
        .minWordFrequency(5)
        .iterations(1)
        .epochs(1)
        .layerSize(50)
        .batchSize(10000)
        .workers(10)
        .seed(42)
        .windowSize(5)
        .iterate(rawDataSetIterator)
        .tokenizerFactory(tokenizerFactory)
        .build()

    logger.info("Starting fitting Word2Vec...")
    vec.fit()

    WordVectorSerializer.writeVocabCache(koreanTfidfVectorizer.vocabCache,File("vocabcache.bin"))
    WordVectorSerializer.writeWord2VecModel(vec,"word2vec.bin")

    var x = mutableListOf<String>()
    var y = mutableListOf<String>()
    for(i in 0 until packedRawDataSet.rawDataSets.size - 1){
        x.add(packedRawDataSet.rawDataSets[i].question!!)
        y.add(packedRawDataSet.rawDataSets[i+1].question!!)
    }


    val rawDataSet = Word2VecRawDataSet(x,y)
    val dataSource = Word2VecDataSource(packedRawDataSet = rawDataSet, word2Vec = vec,koreanTfidfVectorizer = koreanTfidfVectorizer, koreanTokenizerFactory = tokenizerFactory)

    val model = KoreanNeuralNetwork.buildNeuralNetworkLSTM(dataSource.inputColumns(),dataSource.inputColumns())
    model.init()

    model.setListeners(ScoreIterationListener(x.size))
    model.fit(dataSource,10)
    ModelSerializer.writeModel(model,"model.bin",true)
/*
    val batchSize = 10

    val epoch = 5

    val maxLen = 20

    val sentences = packedRawDataSet.dialogues.flatMap { it.rawDataSets }.map { it.question!! }
    var iterator =
        CharacterIterator(sentences, batchSize, maxLen, CharacterIterator.defaultCharacterSet, SecureRandom())
    val model = KoreanNeuralNetwork.buildNeuralNetworkLSTM(iterator.inputColumns(), iterator.totalOutcomes())
    model.init()


    val listener = ScoreIterationListener(1000)
    model.setListeners(listener)
    model.fit(iterator, epoch)

    logger.info("Total score: ${model.score()}")
    ModelSerializer.writeModel(model,"model.lstm",true)
*/
}
private fun sampleCharactersFromNetwork(
    initialization: String, net: MultiLayerNetwork,
    iter: CharacterIterator, rng: Random, charactersToSample: Int, numSamples: Int
): Array<String?>? {
    //Set up initialization. If no initialization: use a random character
    var initialization: String? = initialization
    if (initialization == null) {
        initialization = java.lang.String.valueOf(iter.randomCharacter)
    }

    //Create input for initialization
    val initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization!!.length)
    val init = initialization.toCharArray()
    for (i in init.indices) {
        val idx = iter.convertCharacterToIndex(init[i])
        for (j in 0 until numSamples) {
            initializationInput.putScalar(intArrayOf(j, idx, i), 1.0f)
        }
    }
    val sb = arrayOfNulls<StringBuilder>(numSamples)
    for (i in 0 until numSamples) sb[i] = StringBuilder(initialization)

    //Sample from network (and feed samples back into input) one character at a time (for all samples)
    //Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output
    for (i in 0 until charactersToSample) {
        //Set up next input (single time step) by sampling from previous output
        val nextInput = Nd4j.zeros(numSamples, iter.inputColumns())
        //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
        for (s in 0 until numSamples) {
            val outputProbDistribution = DoubleArray(iter.totalOutcomes())
            for (j in outputProbDistribution.indices) outputProbDistribution[j] =
                output.getDouble(s.toLong(), j.toLong())
            val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
            nextInput.putScalar(intArrayOf(s, sampledCharacterIdx), 1.0f) //Prepare next time step input
            sb[s]!!.append(iter.convertIndexToCharacter(sampledCharacterIdx)) //Add sampled character to StringBuilder (human readable output)
        }
        output = net.rnnTimeStep(nextInput) //Do one time step of forward pass
    }
    val out = arrayOfNulls<String>(numSamples)
    for (i in 0 until numSamples) out[i] = sb[i].toString()
    return out
}

/** Given a probability distribution over discrete classes, sample from the distribution
 * and return the generated class index.
 * @param distribution Probability distribution over classes. Must sum to 1.0
 */
fun sampleFromDistribution(distribution: DoubleArray, rng: Random): Int {
    val d: Double = rng.nextDouble()
    var sum = 0.0
    for (i in distribution.indices) {
        sum += distribution[i]
        if (d <= sum) return i
    }
    throw IllegalArgumentException("Distribution is invalid? d=$d, sum=$sum")
}