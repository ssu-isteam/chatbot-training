package dev.isteam.chatbot.dl.engines


import java.io.File
import java.util.*

internal class QAEngineTest {

    fun initEngine() {
        /*
        var path = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\일반상식\\02_squad_질문_답변_제시문_말뭉치\\ko_wiki_v1_squad.json"
        var path2 = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\기계독해\\기계독해분야\\ko_nia_normal_squad_all.json"
        var loader = MINDsLabDataSetLoader(path)
        var loader2 = MINDsLabDataSetLoader(path2)
        var packedRawDataSet = loader.load().get()
        var packedRawDataSet2 = loader2.load().get()

        packedRawDataSet.rawDataSets.plus(packedRawDataSet2.rawDataSets)

        var tokenizerFactory = KoreanTokenizerFactory()
        tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

        var word2Vec = WordVectorSerializer.readWord2VecModel("word2VecModel.w2v")

        var textConverter = TextConverter(word2Vec,tokenizerFactory)

        var network = ModelSerializer.restoreMultiLayerNetwork("network.model")

        var result = network.output(koreanTfidfVectorizer.transform("다테 기미코가 최초로 은퇴 선언을 한게 언제지"))

        var index = result.argMax(1,1)
        println(index.toStringFull())

         */
    }

}