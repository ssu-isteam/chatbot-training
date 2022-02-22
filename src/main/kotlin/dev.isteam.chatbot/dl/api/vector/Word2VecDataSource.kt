package dev.isteam.chatbot.dl.api.vector

import dev.isteam.chatbot.dl.api.dataset.LSTMPackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import dev.isteam.chatbot.dl.api.preprocessor.MeanEmbeddingVectorizer
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import logger
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import kotlin.math.log


class Word2VecDataSource(
    private val packedRawDataSet: LSTMPackedRawDataSet,
    private val word2Vec: Word2Vec,
    private val koreanTfidfVectorizer: KoreanTfidfVectorizer,
    private val koreanTokenizerFactory: KoreanTokenizerFactory,
    private val meanEmbeddingVectorizer: MeanEmbeddingVectorizer = MeanEmbeddingVectorizer(
        koreanTfidfVectorizer,
        word2Vec
    ),
    private val batchSize: Int = 100,
    private var dataCount: Int = 0,
    private var preProcessor: DataSetPreProcessor? = null,
    private var iterator: ListIterator<PackedRawDataSet> = packedRawDataSet.dialogues.listIterator(),
    private val labels: MutableList<String> = IntRange(
        0,
        packedRawDataSet.rawDataSets.size - 1
    ).map { it.toString() }.toMutableList(),
    private val timeSeries: Int = packedRawDataSet.max,

    var currentCount: Long = 0
) : DataSetIterator {



    /**
     * Removes from the underlying collection the last element returned by this iterator.
     */
    override fun remove() {

    }

    /**
     * Returns `true` if the iteration has more elements.
     */
    override fun hasNext(): Boolean {
        return iterator.hasNext()
    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    override fun next(num: Int): DataSet {
        return nextDataSet(num)
    }

    private fun mdsToDataSet(mds: MultiDataSet): DataSet {
        var f = mds.features[0]
        var fm = mds.featuresMaskArrays[0]

        var l = mds.labels[0]
        var lm = mds.labelsMaskArrays[0]

        return DataSet(f, l, fm, lm)
    }
    /*
    private fun nextMultiDataSet(num: Int) : MultiDataSet{
        var datasetList = ArrayList<DataSet>()
        for(i in 0 until num){
            if(! hasNext())
                break
            var dataSet = nextDataSet()
            datasetList.add(dataSet)
        }
        return MultiDataSet(datasetList.map { dataSet -> dataSet.features}.toTypedArray(),datasetList.map { dataSet -> dataSet.labels }.toTypedArray())
    }
     */
    /**
     * Returns the next element in the iteration.
     */
    override fun next(): DataSet {
        return next(batchSize)
    }


    private fun nextDataSet(numExamples: Int): DataSet {
        var features = Nd4j.create(numExamples, inputColumns(), timeSeries)
        var labelsVector = Nd4j.create(numExamples, timeSeries, timeSeries)
        for (i in 0 until numExamples) {
            if (!hasNext())
                break
            dataCount = 0
            var index = iterator.nextIndex()
            var nextDialogue = iterator.next()


            nextDialogue.rawDataSets.forEachIndexed{ index, dataSet ->
                var tokenizer = koreanTokenizerFactory.create(dataSet.question)
                var tokens = tokenizer.tokens
                var featureVector = meanEmbeddingVectorizer.transform(tokens).toDoubleVector()
                for (j in featureVector.indices)
                    features.putScalar(intArrayOf(i, j, dataCount), featureVector[j])


                var labelVector =
                    toOutcomeVector(index, nextDialogue.rawDataSets.size)
                        .toIntVector()

                for (j in labelVector.indices)
                    labelsVector.putScalar(intArrayOf(i, j, dataCount), labelVector[j])
                dataCount++
            }

        }

        return DataSet(features, labelsVector)
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    override fun inputColumns(): Int {
        return word2Vec.layerSize
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    override fun totalOutcomes(): Int {
        return packedRawDataSet.rawDataSets.size
    }

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    override fun resetSupported(): Boolean {
        return true
    }

    /**
     * Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
     * Most DataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
     * iterator that does asynchronous prefetching. For example, it would not make sense to use asynchronous
     * prefetching for the following types of iterators:
     * (a) Iterators that store their full contents in memory already
     * (b) Iterators that re-use features/labels arrays (as future next() calls will overwrite past contents)
     * (c) Iterators that already implement some level of asynchronous prefetching
     * (d) Iterators that may return different data depending on when the next() method is called
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    override fun asyncSupported(): Boolean {
        return true
    }

    /**
     * Resets the iterator back to the beginning
     */
    override fun reset() {
        dataCount = 0
        iterator = packedRawDataSet.dialogues.listIterator()
    }

    /**
     * Batch size
     *
     * @return
     */
    override fun batch(): Int {
        return batchSize
    }

    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    override fun setPreProcessor(preProcessor: DataSetPreProcessor?) {
        this.preProcessor = preProcessor
    }

    /**
     * Returns preprocessors, if defined
     *
     * @return
     */
    override fun getPreProcessor(): DataSetPreProcessor {
        return preProcessor!!
    }

    /**
     * Get dataset iterator class labels, if any.
     * Note that implementations are not required to implement this, and can simply return null
     */
    override fun getLabels(): MutableList<String> {
        return labels
    }

    private fun toOutcomeVector(idx:Int, size:Int) : INDArray{
        return Nd4j.zeros(1,size).also { it.putScalar(idx.toLong(),1) }
    }
}