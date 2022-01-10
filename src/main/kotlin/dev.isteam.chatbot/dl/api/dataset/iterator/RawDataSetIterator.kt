package dev.isteam.chatbot.dl.api.dataset.iterator

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import org.deeplearning4j.text.sentenceiterator.SentenceIterator
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor

class RawDataSetIterator(
    private val packedRawDataSet: PackedRawDataSet,
    private val iterativeType: IterativeType,
    var iterator: Iterator<String> = if (iterativeType == IterativeType.QUESTION) packedRawDataSet.rawDataSets.map { it.question }
        .iterator() else packedRawDataSet.rawDataSets.map { it.answer }.iterator()
) : SentenceIterator {

    private var preProcessor: SentencePreProcessor? = null

    /**
     * Gets the next sentence or null
     * if there's nothing left (Do yourself a favor and
     * check hasNext() )
     *
     * @return the next sentence in the iterator
     */
    override fun nextSentence(): String {
        return if (preProcessor != null) preProcessor!!.preProcess(iterator.next()) else iterator.next()
    }

    /**
     * Same idea as [java.util.Iterator]
     * @return whether there's anymore sentences left
     */
    override fun hasNext(): Boolean {
        return iterator.hasNext()
    }

    /**
     * Resets the iterator to the beginning
     */
    override fun reset() {
        iterator = if (iterativeType == IterativeType.QUESTION) packedRawDataSet.rawDataSets.map { it.question }
            .iterator() else packedRawDataSet.rawDataSets.map { it.answer }.iterator()
    }

    /**
     * Allows for any finishing (closing of input streams or the like)
     */
    override fun finish() {
        //This only works for file input stream
    }

    override fun getPreProcessor(): SentencePreProcessor {
        return preProcessor!!
    }

    override fun setPreProcessor(preProcessor: SentencePreProcessor?) {
        this.preProcessor = preProcessor
    }

    enum class IterativeType {
        QUESTION,
        ANSWER
    }
}