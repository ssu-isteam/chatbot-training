import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.AutoencoderDataSource
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import me.tongfei.progressbar.ProgressBarBuilder
import me.tongfei.progressbar.ProgressBarStyle
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Paths
import java.time.temporal.ChronoUnit


val logger = LoggerFactory.getLogger("Main")

fun main2(args: Array<String>) {
    val motherPath = ""
    val files =
        arrayOf("개인및관계.json"/*"미용과건강.json","상거래(쇼핑).json","시사교육.json","식음료.json","여가생활.json","일과직업.json","주거와생활.json","행사.json"*/)


    logger.info("Reading files....")

    var viveDataSetLoader = VIVEDataSetLoader(files.map { Paths.get(motherPath, it).toString() }.toTypedArray())

    var packedRawDataSet = viveDataSetLoader.load().get()


    //packedRawDataSet.rawDataSets = ArrayList( packedRawDataSet.rawDataSets.subList(0,5000))

    logger.info("Reading files completed. Total count: ${packedRawDataSet.rawDataSets.size}")


    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var batchSize = 5000

    logger.info("Starting fitting tfidf vectorizer....")

    var koreanTfidfVectorizer =
        KoreanTfidfVectorizer(packedRawDataSet = packedRawDataSet, koreanTokenizerFactory = tokenizerFactory)
    koreanTfidfVectorizer.fit()

    val epoch = 100


    var dataSource =
        AutoencoderDataSource(packedRawDataSet = packedRawDataSet, ktfid = koreanTfidfVectorizer, batchSize = batchSize)

    var network = KoreanNeuralNetwork.buildVariationalAutoEncoder(koreanTfidfVectorizer.nIn())
    network.init()


    var progressBar = ProgressBarBuilder().setTaskName("Fitting autoencoder")
        .setInitialMax(packedRawDataSet.rawDataSets.size.toLong() * epoch)
        .setStyle(ProgressBarStyle.ASCII)
        .setSpeedUnit(ChronoUnit.SECONDS)
        .build()

    progressBar.run {
        for (i in 0 until epoch) {
            while (dataSource.hasNext()) {
                var data = dataSource.next()
                network.fit(data.features, data.features)
                stepBy(dataSource.currentCount)
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
    ModelSerializer.writeModel(network,"autoencoder.model",true)
    logger.info(eval.stats())
}