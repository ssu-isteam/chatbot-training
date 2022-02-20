package dev.isteam.chatbot.dl.api.preprocessor

import dev.isteam.chatbot.dl.api.vector.KoreanTfidfVectorizer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class MeanEmbeddingVectorizer(val tfid: KoreanTfidfVectorizer, val word2Vec: Word2Vec) {
    fun transform(tokens: List<String>): INDArray {
        var stack = Nd4j.create(tokens.size, word2Vec.layerSize)
        for (i in tokens.indices) {
            if(word2Vec.getWordVector(tokens[i]) == null)
                continue
            stack.putRow(i.toLong(), Nd4j.create(word2Vec.getWordVector(tokens[i])))
        }
        return Nd4j.mean(stack,1)
    }

    fun transform(tokens: List<String>, input: INDArray): INDArray {
        var wordVector = input.getRow(0).toDoubleVector()
        var converted = Nd4j.create(1, wordVector.size)
        for (i in wordVector.indices) {
            converted.putScalar(i.toLong(), wordVector[i] * tfid.tfidf(tokens, tokens[i]))
        }
        return converted
    }
}