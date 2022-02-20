package dev.isteam.chatbot.dl.api.vector

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class AutoencoderDataSource(
    private val packedRawDataSet: PackedRawDataSet,
    private val ktfid: KoreanTfidfVectorizer,
    private val batchSize: Int = 100,
    private var preProcessor: DataSetPreProcessor? = null,
    private var iterator: Iterator<RawDataSet> = packedRawDataSet.rawDataSets.iterator(),
    private val labels: MutableList<String> = IntRange(
        0,
        packedRawDataSet.rawDataSets.size - 1
    ).map { it.toString() }.toMutableList(),

    var currentCount: Long = 0,
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
        currentCount = 0
        var features = Nd4j.zeros(num, inputColumns())
        for (i in 0 until num) {
            if (!hasNext())
                break
            var rawDataSet = iterator.next()
            var f = ktfid.transform(rawDataSet.question)
            features.putRow(i.toLong(), f)
            currentCount++
        }
        return DataSet(features, null)
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
        var features = Nd4j.create(numExamples, inputColumns(), packedRawDataSet.rawDataSets.size)
        var labelVector = Nd4j.create(numExamples, inputColumns(), packedRawDataSet.rawDataSets.size)
        var dataCount = 0
        for (i in 0 until numExamples) {
            if (!hasNext())
                break
            var rawDataSet = iterator.next()
            var rawFeatures = ktfid.transform(rawDataSet.question).toDoubleVector()

            for (j in rawFeatures.indices) {
                features.putScalar(intArrayOf(i, j, dataCount), rawFeatures[j])
            }
            if (!hasNext())
                break

            var nextDataSet = iterator.next()
            var nextFeatures = ktfid.transform(nextDataSet.question).toDoubleVector()
            for (j in nextFeatures.indices) {
                labelVector.putScalar(intArrayOf(i, j, dataCount), rawFeatures[j])
            }
            dataCount++
        }
        return DataSet(features, labelVector)
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    override fun inputColumns(): Int {
        return ktfid.nIn()
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
        iterator = packedRawDataSet.rawDataSets.iterator()
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


}