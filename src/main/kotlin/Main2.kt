
import dev.isteam.chatbot.dl.api.dataset.iterator.CharacterIterator
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.optimize.listeners.TimeIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.nio.file.Paths
import java.security.SecureRandom


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

    /*
    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var batchSize = 1000

    logger.info("Starting fitting tfidf vectorizer....")

    var vocabCache = WordVectorSerializer.readVocabCache(File("vocab.cache"))



    var koreanTfidfVectorizer =
        KoreanTfidfVectorizer(packedRawDataSet = packedRawDataSet, koreanTokenizerFactory = tokenizerFactory, cache = vocabCache)
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
*/

    val batchSize = 100

    val epoch = 50

    val maxLen = 20

    val sentences = packedRawDataSet.dialogues.flatMap { it.rawDataSets }.map { it.question!! }
    var iterator =
        CharacterIterator(sentences, batchSize, maxLen, CharacterIterator.defaultCharacterSet, SecureRandom())
    val model = KoreanNeuralNetwork.buildNeuralNetworkLSTM(iterator.inputColumns(), iterator.totalOutcomes())

    val listener = ScoreIterationListener(iterator.totalExamples())
    model.setListeners(listener)
    model.fit(iterator, epoch)
    logger.info("Total score: ${model.score()}")
    ModelSerializer.writeModel(model,"model.lstm",true)
}