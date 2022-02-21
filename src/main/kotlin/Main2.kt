import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import dev.isteam.chatbot.dl.api.dataset.iterator.RawDataSetIterator
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.AutoencoderDataSource
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.api.vector.Word2VecDataSource
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Paths
import java.time.temporal.ChronoUnit


val logger: Logger = LoggerFactory.getLogger("Main")

fun main2(args: Array<String>) {
    val motherPath = "D:\\ISTEAM\\AI 데이터 셋\\한국어 대화 요약\\Training\\[라벨]한국어대화요약_train"
    val files =
        arrayOf("개인및관계.json"/*"미용과건강.json","상거래(쇼핑).json","시사교육.json","식음료.json","여가생활.json","일과직업.json","주거와생활.json","행사.json"*/)

    logger.info("Reading files....")

    var viveDataSetLoader = VIVEDataSetLoader(files.map { Paths.get(motherPath, it).toString() }.toTypedArray())

    var packedRawDataSet = viveDataSetLoader.load().get()

    packedRawDataSet.rawDataSets = packedRawDataSet.rawDataSets.subList(0,500)
    packedRawDataSet.rawDataSets = filter(packedRawDataSet,true)

    logger.info("Reading files completed. Total count: ${packedRawDataSet.rawDataSets.size}")


    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var batchSize = 100

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
        .seed(42)
        .windowSize(5)
        .iterate(rawDataSetIterator)
        .tokenizerFactory(tokenizerFactory)
        .build()

    logger.info("Starting fitting Word2Vec...")
    vec.fit()


    val epoch = 100


    var dataSource = Word2VecDataSource(packedRawDataSet = packedRawDataSet, word2Vec = vec, koreanTfidfVectorizer = koreanTfidfVectorizer, koreanTokenizerFactory = tokenizerFactory, batchSize = batchSize)

    logger.info("Initializing network...")

    var network = KoreanNeuralNetwork.buildNeuralNetworkLSTM(vec.layerSize, packedRawDataSet.rawDataSets.size)
    network.init()


    var progressBar = ProgressBarBuilder().setTaskName("Fitting network")
        .setInitialMax(packedRawDataSet.rawDataSets.size.toLong() * epoch)
        .setStyle(ProgressBarStyle.ASCII)
        .setSpeedUnit(ChronoUnit.SECONDS)
        .build()

    progressBar.run {
        for (i in 0 until epoch) {
            while (dataSource.hasNext()) {
                var data = dataSource.next()
                network.fit(data.features, data.labels)
         //       stepBy(dataSource.currentCount)
            }
            logger.info("${(i/ epoch.toDouble()) * 100}%")
            dataSource.reset()
        }
    }


    var eval = Evaluation(packedRawDataSet.rawDataSets.size)

    while (dataSource.hasNext()) {
        var data = dataSource.next()
        var output = network.output(data.features)
        //  logger.info("eval: predicted: $output , real: ${data.features}")
        eval.eval(data.features, output)
    }
    WordVectorSerializer.writeVocabCache(koreanTfidfVectorizer.vocabCache,File("vocab.cache"))
    ModelSerializer.writeModel(network,"network.model",true)
    logger.info(eval.stats())
}
fun filter(packedRawDataSet: PackedRawDataSet, even:Boolean) : List<RawDataSet> {
    return if(even) packedRawDataSet.rawDataSets.filterIndexed { index, _ ->  index % 2 == 0 } else packedRawDataSet.rawDataSets.filterIndexed { index, _ -> index % 2 != 0 }
}
