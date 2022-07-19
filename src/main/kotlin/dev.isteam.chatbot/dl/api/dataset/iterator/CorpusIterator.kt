package dev.isteam.chatbot.dl.api.dataset.iterator

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import org.apache.commons.lang3.ArrayUtils
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.ceil

class CorpusIterator(
    private val word2Vec: Word2Vec,
    private val corpus: PackedRawDataSet,
    private val tokenizerFactory: KoreanTokenizerFactory,
    private val batchSize: Int,
    private val batchesPerMacroBatch: Int,
    private val maxLenPerSentence: Int,
) : MultiDataSetIterator {
    private val dictSize = word2Vec.vocab.numWords()
    private val totalBatches = ceil(corpus.size().toDouble() / batchSize).toInt()
    private val totalMacroBatches = ceil(totalBatches.toDouble() / batchesPerMacroBatch).toInt()
    private var currentBatch = 0
    private var currentMacroBatch = 0
    override fun remove() {

    }

    override fun hasNext(): Boolean {
        return currentBatch < totalBatches && getMacroBatchByCurrentBatch() == currentMacroBatch
    }

    private fun getMacroBatchByCurrentBatch(): Int {
        return currentBatch / batchesPerMacroBatch
    }

    override fun next(num: Int): MultiDataSet {
        var i = currentBatch * batchSize
        var currentBatchSize = batchSize.coerceAtMost(corpus.size() - i - 1)
        val input = Nd4j.zeros(currentBatchSize, 1, maxLenPerSentence)
        val prediction = Nd4j.zeros(currentBatchSize, dictSize, maxLenPerSentence)
        val decode = Nd4j.zeros(currentBatchSize, dictSize, maxLenPerSentence)
        val inputMask = Nd4j.zeros(currentBatchSize, maxLenPerSentence)
        val predictionMask = Nd4j.zeros(currentBatchSize, maxLenPerSentence)



        for (j in 0 until currentBatchSize) {
            val rowIn = getIndices(corpus[i]).reversed()
            val rowPred = getIndices(corpus[i + 1])

            rowPred.add(1.0)

            input.put(
                arrayOf<INDArrayIndex>(
                    NDArrayIndex.point(j.toLong()),
                    NDArrayIndex.point(0),
                    NDArrayIndex.interval(0, rowIn.size)
                ),
                Nd4j.create(ArrayUtils.toPrimitive(rowIn.toTypedArray()))
            )

            inputMask.put(
                arrayOf(NDArrayIndex.point(j.toLong()), NDArrayIndex.interval(0, rowIn.size)),
                Nd4j.ones(rowIn.size)
            )
            predictionMask.put(
                arrayOf(NDArrayIndex.point(j.toLong()), NDArrayIndex.interval(0, rowPred.size)),
                Nd4j.ones(rowPred.size)
            )

            val predOneHot = Array(dictSize) { DoubleArray(rowPred.size) { 0.0 } }
            val decodeOneHot = Array(dictSize) { DoubleArray(rowPred.size) { 0.0 } }

            decodeOneHot[2][0] = 1.0

            rowPred.forEachIndexed { predIdx, pred ->
                if (pred.toInt() >= dictSize)
                    return@forEachIndexed
                predOneHot[pred.toInt()][predIdx] = 1.0
                if (predIdx < rowPred.size - 1) {
                    decodeOneHot[pred.toInt()][predIdx + 1] = 1.0
                }
            }
            prediction.put(
                arrayOf(
                    NDArrayIndex.point(j.toLong()), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size)
                ), Nd4j.create(predOneHot)
            )
            decode.put(
                arrayOf(
                    NDArrayIndex.point(j.toLong()), NDArrayIndex.interval(0, dictSize),
                    NDArrayIndex.interval(0, rowPred.size)
                ), Nd4j.create(decodeOneHot)
            )


        }
        return org.nd4j.linalg.dataset.MultiDataSet(
            arrayOf(input, decode),
            arrayOf(prediction),
            arrayOf(inputMask, predictionMask),
            arrayOf(predictionMask)
        )
    }

    private fun getIndices(text: String): MutableList<Double> {
        val tokens = tokenizerFactory.create(text).tokens
        return tokens.map { token -> word2Vec.vocab.wordFrequency(token).toDouble() }.toMutableList()
    }

    override fun next(): MultiDataSet {
        return next(batchSize)
    }

    override fun setPreProcessor(preProcessor: MultiDataSetPreProcessor?) {

    }

    override fun getPreProcessor(): MultiDataSetPreProcessor? {
        return null
    }

    override fun resetSupported(): Boolean {
        return false
    }

    override fun asyncSupported(): Boolean {
        return true
    }

    override fun reset() {
        currentBatch = 0
        currentMacroBatch = 0
    }

    fun batch(): Int {
        return currentBatch
    }

    fun totalBatches(): Int {
        return totalBatches
    }

    fun setCurrentBatch(currentBatch: Int) {
        this.currentBatch = currentBatch
        currentMacroBatch = getMacroBatchByCurrentBatch()
    }

    fun hasNextMacroBatch(): Boolean {
        return getMacroBatchByCurrentBatch() < totalMacroBatches && currentMacroBatch < totalMacroBatches
    }

    fun nextMacroBatch() {
        ++currentMacroBatch
    }

}