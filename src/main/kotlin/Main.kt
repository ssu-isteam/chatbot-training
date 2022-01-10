import dev.isteam.chatbot.dl.api.dataset.loader.MINDsLabDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import dev.isteam.chatbot.dl.api.vector.DataSource
import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import dev.isteam.chatbot.dl.engines.KoreanNeuralNetwork
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import java.io.File

fun main(args: Array<String>) {
    var path = "ko_wiki_v1_squad.json"
    var path2 = "ko_nia_normal_squad_all.json"
    var loader = MINDsLabDataSetLoader(path)
    var loader2 = MINDsLabDataSetLoader(path2)
    var packedRawDataSet = loader.load().get()
    var packedRawDataSet2 = loader2.load().get()

    packedRawDataSet.rawDataSets.plus(packedRawDataSet2.rawDataSets)

    var tokenizerFactory = KoreanTokenizerFactory()
    tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

    var word2Vec = WordVectorSerializer.readWord2VecModel("word2Vec.w2v")

    var batchSize = 10

    var dataSource = DataSource(packedRawDataSet = packedRawDataSet, word2Vec = word2Vec, batchSize = batchSize, koreanTokenizerFactory = tokenizerFactory)

    var network = KoreanNeuralNetwork.buildNeuralNetworkLSTM(word2Vec.layerSize,packedRawDataSet.rawDataSets.size)

    network.init()
    network.setListeners(ScoreIterationListener(1))
    for(i in 0 until 10){
        while(dataSource.hasNext()){
            var dataSet = dataSource.next()
            network.fit(dataSet)
        }
        println("Current epoch: $i")
        dataSource.reset()
    }
    var eval = Evaluation(packedRawDataSet.rawDataSets.size)
    while (dataSource.hasNext()){
        var t = dataSource.next();
        var features = t.features
        var labels = t.labels
        var predicted = network.output(features,false)
        eval.eval(labels,predicted)
    }

    println(eval.stats())

    ModelSerializer.writeModel(network,"network.model",true)
}