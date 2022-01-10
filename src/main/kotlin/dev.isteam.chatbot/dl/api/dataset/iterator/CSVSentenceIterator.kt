package dev.isteam.chatbot.dl.api.dataset.iterator

import org.deeplearning4j.text.sentenceiterator.SentenceIterator
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader

class CSVSentenceIterator(input: File?, targetIndex: Int) : SentenceIterator {
    private var sentencePreProcessor: SentencePreProcessor? = null
    private val reader: BufferedReader
    private val lines: MutableList<String>
    private var iter: Iterator<String>
    private var targetIndex = 0
    override fun nextSentence(): String {
        return iter.next().split(",").toTypedArray()[targetIndex].trim { it <= ' ' }
    }

    override fun hasNext(): Boolean {
        return iter.hasNext()
    }

    override fun reset() {
        iter = lines.iterator()
    }

    override fun finish() {
        iter = lines.iterator()
    }

    override fun getPreProcessor(): SentencePreProcessor {
        return sentencePreProcessor!!
    }

    override fun setPreProcessor(sentencePreProcessor: SentencePreProcessor) {
        this.sentencePreProcessor = sentencePreProcessor
    }

    fun setTargetIndex(targetIndex: Int) {
        this.targetIndex = targetIndex
    }

    init {
        reader = BufferedReader(InputStreamReader(FileInputStream(input), "euc-kr"))
        this.targetIndex = targetIndex
        lines = ArrayList()
        reader.readLine() //Remove first line
        var line: String
        while (reader.readLine().also { line = it } != null) lines.add(line)
        iter = lines.iterator()
        reader.close()
    }
}