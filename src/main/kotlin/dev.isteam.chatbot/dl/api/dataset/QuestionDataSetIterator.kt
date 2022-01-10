package dev.isteam.chatbot.dl.api.dataset

import au.com.bytecode.opencsv.CSVReader
import dev.isteam.chatbot.dl.api.dataset.iterator.CSVSentenceIterator
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer
import org.deeplearning4j.text.documentiterator.LabelsSource
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.util.FeatureUtil
import org.nd4j.shade.guava.base.Functions
import org.nd4j.shade.guava.collect.Lists
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.util.stream.Collectors
import java.util.stream.IntStream

class QuestionDataSetIterator private constructor(input: File, converter: TfidfVectorizer?, answersCount: Int) :
    DataSetIterator {
    private val reader: CSVReader
    private var converter: TfidfVectorizer? = null
    private var nOut: Int
    private var nIn = 0
    private var questionIter: Iterator<String?>? = null
    private val answers: HashMap<Int, String?>
    private var labels: List<String> = ArrayList()
    private var source: LabelsSource? = null
    private var preProcessor: DataSetPreProcessor? = null
    private var pos = 0

    /***
     *
     * @param input 읽어들일 CSV 파일
     * @param labelIndex CSV 파일을 읽어서 , 로 split 했을 때 label의 position
     * @param questionIndex CSV 파일 형식에서 , 로 split 했을 때 question의 position
     * @throws IOException
     * @throws InterruptedException
     */
    private constructor(input: File, labelIndex: Int, questionIndex: Int) : this(input, null, 0) {
        //CSV 파일 맨 윗줄에 있는 레이블을 제거합니다.
        reader.readNext()
        var row: Array<String?>
        while (reader.readNext().also { row = it } != null) {
            answers[nOut] = row[questionIndex]
            nOut++
        }
        source = LabelsSource(
            Lists.transform(
                IntStream.range(0, nOut).boxed().collect(Collectors.toList()),
                Functions.toStringFunction()
            )
        )
        questionIter = answers.values.iterator()
        converter = TfidfVectorizer.Builder()
            .setIterator(CSVSentenceIterator(input, questionIndex))
            .setTokenizerFactory(DefaultTokenizerFactory())
            .build()

        converter!!.fit()
        //입력 노드의 수를 정합니다.
        labels = IntStream.range(0, nOut).boxed().map { i: Int? ->
            (i!!).toString()
        }.collect(Collectors.toList())
    }

    override fun next(i: Int): DataSet {
        val text = questionIter!!.next()
        val features = converter!!.transform(text)
        val labels = features.getRow(0).putScalar(0, 1.0)
        return DataSet(features, labels)
    }

    /***
     *
     * @return 입력 노드의 개수
     */
    override fun inputColumns(): Int {
        return nIn
    }

    /***
     *
     * @return 출력 노드의 개수
     */
    override fun totalOutcomes(): Int {
        return nOut
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        return false
    }

    override fun reset() {
        pos = 0
        questionIter = answers.values.iterator()
    }

    override fun batch(): Int {
        return 0
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        return preProcessor!!
    }

    override fun setPreProcessor(dataSetPreProcessor: DataSetPreProcessor) {
        preProcessor = dataSetPreProcessor
    }

    /***
     *
     * @return 레이블
     */
    override fun getLabels(): List<String> {
        return labels
    }

    /**
     * Removes from the underlying collection the last element returned by this iterator.
     */
    override fun remove() {
    }

    /**
     * Returns `true` if the iteration has more elements.
     * (In other words, returns `true` if [.next] would
     * return an element rather than throwing an exception.)
     *
     * @return `true` if the iteration has more elements
     */
    override fun hasNext(): Boolean {
        return questionIter!!.hasNext()
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next dataset in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Throws(NoSuchElementException::class)
    override fun next(): DataSet {
        val q = questionIter!!.next()
        val feature = converter!!.transform(q)
        val label =
            FeatureUtil.toOutcomeVector(source!!.indexOf(Integer.toString(pos)).toLong(), source!!.size().toLong())
        val dataSet = DataSet(feature, label)
        pos++
        return dataSet
    }

    companion object {
        /***
         *
         * @param input 읽어들일 CSV 파일
         * @param labelIndex CSV 파일을 읽어서 , 로 split 했을 때 label의 position
         * @param questionIndex CSV 파일 형식에서 , 로 split 했을 때 question의 position
         * @return 새로운 QuestionDataSetIterator 인스턴스
         * @throws IOException
         * @throws InterruptedException
         */
        @Throws(IOException::class, InterruptedException::class)
        fun newIterator(input: File, labelIndex: Int, questionIndex: Int): QuestionDataSetIterator {
            return QuestionDataSetIterator(input, labelIndex, questionIndex)
        }
    }

    init {
        reader = CSVReader(InputStreamReader(FileInputStream(input), StandardCharsets.UTF_8))
        this.converter = converter
        nOut = answersCount
        answers = HashMap()
    }
}