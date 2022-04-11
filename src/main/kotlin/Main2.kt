import dev.isteam.chatbot.dl.api.dataset.Word2VecRawDataSet
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.api.vector.Word2VecDataSource
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.classification.ROC
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Paths


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
    /*
    var koreanTfidfVectorizer =
        KoreanTfidfVectorizer(packedRawDataSet = packedRawDataSet, koreanTokenizerFactory = tokenizerFactory)
    koreanTfidfVectorizer.fit()


    var rawDataSetIterator = RawDataSetIterator(packedRawDataSet,RawDataSetIterator.IterativeType.QUESTION)


    val vec: Word2Vec = Word2Vec.Builder()
        .minWordFrequency(5)
        .iterations(1)
        .epochs(3)
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

     */
    val vec = WordVectorSerializer.readWord2VecModel(File("word2vec.bin"), true)
    val vocabCache = WordVectorSerializer.readVocabCache(File("vocabcache.bin"))
    val koreanTfidfVectorizer = KoreanTfidfVectorizer(
        packedRawDataSet = packedRawDataSet,
        koreanTokenizerFactory = tokenizerFactory,
        cache = vocabCache
    )
    var x = mutableListOf<String>()
    var y = mutableListOf<String>()
    for (i in 0 until packedRawDataSet.rawDataSets.size - 1) {
        x.add(packedRawDataSet.rawDataSets[i].question!!)
        y.add(packedRawDataSet.rawDataSets[i + 1].question!!)
    }


    val rawDataSet = Word2VecRawDataSet(x, y)
    val dataSource = Word2VecDataSource(
        packedRawDataSet = rawDataSet,
        word2Vec = vec,
        koreanTfidfVectorizer = koreanTfidfVectorizer,
        koreanTokenizerFactory = tokenizerFactory,
        batchSize = batchSize
    )

    val model = KoreanNeuralNetwork.buildSeq2Seq(dataSource.inputColumns(),vec.layerSize)
    model.init()
    model.setListeners(EvaluativeListener(dataSource,1,InvocationType.EPOCH_END))
    model.fit(dataSource,50)
    ModelSerializer.writeModel(model, "model.bin", true)
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