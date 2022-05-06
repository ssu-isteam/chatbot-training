package dev.isteam.chatbot.dl.api.vector

import dev.isteam.chatbot.dl.api.preprocessor.MeanEmbeddingVectorizer
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j

class Seq2SeqDataSource(private val corpus:List<KoreanTokenizer>,private val meanEmbeddingVectorizer: MeanEmbeddingVectorizer,private val batchSize:Int,private val batchesPerMacroBatch:Int, private val maxTokenLen:Int) : MultiDataSetIterator {
    private var currentBatch:Int = 0
    override fun remove() {
        TODO("Not yet implemented")
    }

    override fun hasNext(): Boolean {
        TODO("Not yet implemented")
    }

    override fun next(num: Int): MultiDataSet {
        val i: Int = currentBatch * batchSize
        val currentBatchSize: Int = batchSize.coerceAtMost(corpus.size - i - 1)
        return org.nd4j.linalg.dataset.MultiDataSet()
    }

    override fun next(): MultiDataSet {
        TODO("Not yet implemented")
    }

    override fun setPreProcessor(preProcessor: MultiDataSetPreProcessor?) {
        TODO("Not yet implemented")
    }

    override fun getPreProcessor(): MultiDataSetPreProcessor {
        TODO("Not yet implemented")
    }

    override fun resetSupported(): Boolean {
        TODO("Not yet implemented")
    }

    override fun asyncSupported(): Boolean {
        TODO("Not yet implemented")
    }

    override fun reset() {
        TODO("Not yet implemented")
    }

}