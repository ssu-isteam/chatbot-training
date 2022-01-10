package dev.isteam.chatbot.dl.api.preprocessor

import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class TextConverter(private val word2Vec: Word2Vec, private val tokenizerFactory: KoreanTokenizerFactory) {
    fun convert(text: String): INDArray {
        var tokenizer = tokenizerFactory.create(text)
        var inputVector = Nd4j.create(100, 100)
        for (i in 0 until tokenizer.tokens.size) {
            var vector: DoubleArray? = word2Vec.getWordVector(tokenizer.tokens[i]) ?: continue
            inputVector.putRow(i.toLong(), Nd4j.create(vector))
        }

        return inputVector
    }
}