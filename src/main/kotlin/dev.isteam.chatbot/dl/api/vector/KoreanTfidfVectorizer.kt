package dev.isteam.chatbot.dl.api.vector

import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.iterator.RawDataSetIterator
import dev.isteam.chatbot.dl.api.tokenizer.KoreanTokenizerFactory
import org.deeplearning4j.bagofwords.vectorizer.BaseTextVectorizer
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
import org.deeplearning4j.text.documentiterator.LabelsSource
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter
import org.nd4j.common.util.MathUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import java.io.File
import java.io.InputStream
import java.nio.charset.Charset
import java.util.concurrent.atomic.AtomicLong


class KoreanTfidfVectorizer(
    private val packedRawDataSet: PackedRawDataSet,
    koreanTokenizerFactory: KoreanTokenizerFactory,
    rawDataSetIterator: RawDataSetIterator = RawDataSetIterator(
        packedRawDataSet = packedRawDataSet,
        iterativeType = RawDataSetIterator.IterativeType.QUESTION
    ),
    _minWordFrequency: Int = 5,
    cache: VocabCache<VocabWord> = AbstractCache.Builder<VocabWord>().build(),
    _stopWords: Collection<String> = ArrayList(),
    _isParallel: Boolean = true,
    labels: LabelsSource = LabelsSource(
        IntRange(0, packedRawDataSet.rawDataSets.size - 1).map {
            it.toString()
        }.toMutableList()
    )
) : BaseTextVectorizer() {
    init {
        tokenizerFactory = koreanTokenizerFactory
        minWordFrequency = _minWordFrequency
        isParallel = _isParallel
        vocabCache = cache
        stopWords = _stopWords
        labelsSource = labels
        iterator = SentenceIteratorConverter(rawDataSetIterator, labelsSource)
    }

    /**
     * Text coming from an input stream considered as one document
     * @param is the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    override fun vectorize(`is`: InputStream?, label: String?): DataSet {
        var text = `is`!!.reader(Charset.forName("UTF-8")).readText()
        return vectorize(text, label)
    }

    /**
     * Vectorizes the passed in text treating it as one document
     * @param text the text to vectorize
     * @param label the label of the text
     * @return a dataset with a transform of weights(relative to impl; could be word counts or tfidf scores)
     */
    override fun vectorize(text: String?, label: String?): DataSet {
        var input = transform(text)
        var labelMatrix = FeatureUtil.toOutcomeVector(
            labelsSource.indexOf(label!!).toLong(),
            labelsSource.size().toLong()
        )
        return DataSet(input, labelMatrix)
    }

    /**
     *
     * @param input the text to vectorize
     * @param label the label of the text
     * @return [DataSet] with a applyTransformToDestination of
     * weights(relative to impl; could be word counts or tfidf scores)
     */
    override fun vectorize(input: File?, label: String?): DataSet {
        var text = input!!.readText(Charset.forName("UTF-8"))
        return vectorize(text, label)
    }

    /**
     * Vectorizes the input source in to a dataset
     * @return Adam Gibson
     */
    override fun vectorize(): DataSet {
        return DataSet()
    }

    /**
     * Transforms the matrix
     * @param text text to transform
     * @return [INDArray]
     */
    override fun transform(text: String?): INDArray {
        var tokenizer = tokenizerFactory.create(text)
        var tokens = tokenizer.tokens
        return transform(tokens)
    }

    /**
     * Transforms the matrix
     * @param tokens
     * @return
     */
    override fun transform(tokens: MutableList<String>?): INDArray {
        val ret = Nd4j.create(1, vocabCache.numWords())

        val counts: MutableMap<String, AtomicLong> = HashMap()
        for (token in tokens!!) {
            if (!counts.containsKey(token)) counts[token] = AtomicLong(0)
            counts[token]!!.incrementAndGet()
        }

        for (i in tokens.indices) {
            val idx = vocabCache.indexOf(tokens[i])
            if (idx >= 0) {
                val tf_idf: Double = tfidfWord(tokens[i], counts[tokens[i]]!!.toLong(), tokens.size.toLong())
                //log.info("TF-IDF for word: {} -> {} / {} => {}", tokens.get(i), counts.get(tokens.get(i)).longValue(), tokens.size(), tf_idf);
                ret.putScalar(idx.toLong(), tf_idf)
            }
        }
        return ret
    }

    fun tfidfWord(word: String, wordCount: Long, documentLength: Long): Double {
        return MathUtils.tfidf(tfForWord(wordCount, documentLength), idfForWord(word))
    }

    private fun tfForWord(wordCount: Long, documentLength: Long): Double {
        return wordCount.toDouble() / documentLength.toDouble()
    }

    private fun idfForWord(word: String): Double {
        return MathUtils.idf(vocabCache.totalNumberOfDocs().toDouble(), vocabCache.docAppearedIn(word).toDouble())
    }
}