package dev.isteam.chatbot.dl.api.vector

import dev.isteam.chatbot.dl.api.dataset.Word2VecRawDataSet
import dev.isteam.chatbot.dl.api.preprocessor.MeanEmbeddingVectorizer
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.ceil


class Word2VecDataSource(
    private val packedRawDataSet: Word2VecRawDataSet,
    private val word2Vec: Word2Vec,
    private val koreanTfidfVectorizer: KoreanTfidfVectorizer,
    private val koreanTokenizerFactory: KoreanTokenizerFactory,
    private var sentencesLeft:Int = packedRawDataSet.x.size,
    private val batchSize: Int = 100,
    private var preProcessor: DataSetPreProcessor? = null,
    private val maxLen:Int = 50,
    private val totalBatches: Int = ceil(packedRawDataSet.x.size.toDouble()/ batchSize).toInt()
) : DataSetIterator {

    private var currentBatch = 0


    /**
     * Removes from the underlying collection the last element returned by this iterator.
     */
    override fun remove() {

    }

    /**
     * Returns `true` if the iteration has more elements.
     */
    override fun hasNext(): Boolean {

        return currentBatch < totalBatches
    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    override fun next(num: Int): DataSet {

        var i = currentBatch * batchSize
        var currentBatchSize = batchSize.coerceAtMost(packedRawDataSet.x.size - i - 1)
        var features = Nd4j.zeros(num,maxLen,word2Vec.layerSize)
        var labels = Nd4j.zeros(num,maxLen,word2Vec.layerSize)
        for(j in 0 until currentBatchSize){
            if(! hasNext()) break
            var sentenceVec = Nd4j.create(maxLen,word2Vec.layerSize)

            var tokenizer = koreanTokenizerFactory.create(packedRawDataSet.x[i])

            val xTokens = tokenizer.tokens

            for(k in 0 until xTokens.size){
                if(k >= maxLen)
                    break
                if(! word2Vec.hasWord(xTokens[k]))
                    continue
                val wordVector = word2Vec.getWordVectorMatrix(xTokens[k])
                sentenceVec.putRow(k.toLong(),wordVector)
            }
            features.putRow(i.toLong(),sentenceVec)

            tokenizer = koreanTokenizerFactory.create(packedRawDataSet.y[i])
            val yTokens = tokenizer.tokens

            var ySentencesVec = Nd4j.create(maxLen,word2Vec.layerSize)

            for(k in 0 until yTokens.size){
                if(k >= maxLen)
                    break
                if(! word2Vec.hasWord(yTokens[k]))
                    continue
                val wordVector = word2Vec.getWordVectorMatrix(yTokens[k])
                ySentencesVec.putRow(k.toLong(),wordVector)

            }

            labels.putRow(i.toLong(),ySentencesVec)
            currentBatch++;
        }
        return DataSet(features,labels)
    }

    /*
    private fun mdsToDataSet(mds: MultiDataSet): DataSet {
        var f = mds.features[0]
        var fm = mds.featuresMaskArrays[0]

        var l = mds.labels[0]
        var lm = mds.labelsMaskArrays[0]

        return DataSet(f, l, fm, lm)
    }

     */
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

    /*
    private fun nextDataSet(numExamples: Int): DataSet {

        var features = Nd4j.create(numExamples, inputColumns(), timeSeries)
        //Last cell is for dialogue index
        var labelsVector = Nd4j.create(numExamples, timeSeries + 1, timeSeries)
        for (i in 0 until numExamples) {
            if (!hasNext())
                break
            dataCount = 0
            var dialogueIdx = iterator.nextIndex()
            var nextDialogue = iterator.next()


            nextDialogue.rawDataSets.forEachIndexed{ index, dataSet ->
                var tokenizer = koreanTokenizerFactory.create(dataSet.question)
                var tokens = tokenizer.tokens
                var featureVector = meanEmbeddingVectorizer.transform(tokens).toDoubleVector()

                for (j in featureVector.indices)
                    features.putScalar(intArrayOf(i, if(j >= 50) 49 else j, dataCount), featureVector[j])


                var labelVector =
                    toOutcomeVector(index,timeSeries)
                        .toIntVector()

                for (j in labelVector.indices)
                    labelsVector.putScalar(intArrayOf(i, j, dataCount), labelVector[j])


                labelsVector.putScalar(intArrayOf(i,timeSeries,dataCount),dialogueIdx / parentExp)
                dataCount++
            }

        }

        return DataSet(features, labelsVector)
    }
    */
    /**
     * Input columns for the dataset
     *
     * @return
     */
    override fun inputColumns(): Int {
        return maxLen
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    override fun totalOutcomes(): Int {
        return maxLen
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
        return false
    }

    /**
     * Resets the iterator back to the beginning
     */
    override fun reset() {
        sentencesLeft = packedRawDataSet.x.size
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
    override fun getLabels(): MutableList<String>? {
        return null
    }

    private fun toOutcomeVector(idx:Int, size:Int) : INDArray{
        return Nd4j.zeros(1,size).also { it.putScalar(idx.toLong(),1) }
    }
}