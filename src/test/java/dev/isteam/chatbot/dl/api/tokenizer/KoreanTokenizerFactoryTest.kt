package dev.isteam.chatbot.dl.api.tokenizer

import dev.isteam.chatbot.dl.api.dataset.loader.MINDsLabDataSetLoader
import dev.isteam.chatbot.dl.api.dataset.preprocessor.KoreanTokenPreprocessor
import dev.isteam.chatbot.dl.api.preprocessor.TextConverter
import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL
import kr.co.shineware.nlp.komoran.core.Komoran
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.util.ModelSerializer
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import java.util.*

internal class KoreanTokenizerFactoryTest {

    @Test
    fun create() {
        /**
        var network = ModelSerializer.restoreMultiLayerNetwork("network.model")

        var word2Vec = WordVectorSerializer.readWord2VecModel("word2Vec.w2v")

        var testText = "다테 기미코가 최초로 은퇴 선언을 한게 언제지"

        var tokenizerFactory = KoreanTokenizerFactory()
        tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

        var textConverter = TextConverter(word2Vec, tokenizerFactory)

        var input = textConverter.convert(testText)

        var output = network.predict(input)
        var answer = output[0]
        var path = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\일반상식\\02_squad_질문_답변_제시문_말뭉치\\ko_wiki_v1_squad.json"
        var path2 = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\기계독해\\기계독해분야\\ko_nia_normal_squad_all.json"
        var loader = MINDsLabDataSetLoader(path)
        var loader2 = MINDsLabDataSetLoader(path2)
        var packedRawDataSet = loader.load().get()
        var packedRawDataSet2 = loader2.load().get()

        packedRawDataSet.rawDataSets.plus(packedRawDataSet2.rawDataSets)

        var answerText = packedRawDataSet.rawDataSets[answer].answer

        println(answerText)
        **/
    }
}