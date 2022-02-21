import dev.isteam.chatbot.dl.api.dataset.loader.MINDsLabDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.loader.VIVEDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.DataSource
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import me.tongfei.progressbar.ProgressBar
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.util.ModelSerializer
import java.io.File
import java.nio.file.Paths

fun main(args: Array<String>) {
    main2(args)
    return
    val motherPath = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\한국어 대화 요약\\Training\\[라벨]한국어대화요약_train"
    val files = arrayOf("개인및관계.json","미용과건강.json","상거래(쇼핑).json","시사교육.json","식음료.json","여가생활.json","일과직업.json","주거와생활.json","행사.json")


    println("Reading files....")

    var viveDataSetLoader = VIVEDataSetLoader(files.map { Paths.get(motherPath,it).toString() }.toTypedArray())

    var packedRawDataSet = viveDataSetLoader.load().get()

    println("Reading files completed. Total count: ${packedRawDataSet.rawDataSets.size}")


    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var batchSize = 100

    println("Starting fitting tfidf vectorizer....")

    var koreanTfidfVectorizer =
        KoreanTfidfVectorizer(packedRawDataSet = packedRawDataSet, koreanTokenizerFactory = tokenizerFactory)
    koreanTfidfVectorizer.fit()

    println("Fitting tfidf vectorizer completed.")

    var dataSource =
        DataSource(packedRawDataSet = packedRawDataSet, ktfid = koreanTfidfVectorizer, batchSize = batchSize)

    println("Preparing lstm network...")

    var network = KoreanNeuralNetwork.buildNeuralNetworkLSTM(koreanTfidfVectorizer.nIn(),koreanTfidfVectorizer.nIn())
    network.init()

    println("Preparing lstm network completed.")

    val epoch = 100

    var progressBar = ProgressBar("Fitting network",packedRawDataSet.rawDataSets.size.toLong()*epoch)

    for(i in 0 until epoch){
        while (dataSource.hasNext()){
            var dataSet = dataSource.next()
            network.fit(dataSet)
            progressBar.step()
        }
    }

}