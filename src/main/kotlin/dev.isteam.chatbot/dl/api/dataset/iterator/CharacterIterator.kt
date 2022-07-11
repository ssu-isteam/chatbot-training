package dev.isteam.chatbot.dl.api.dataset.iterator

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.nio.charset.Charset
import java.io.IOException
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import java.io.File
import java.lang.UnsupportedOperationException
import java.nio.file.Files
import java.util.*
import dev.isteam.chatbot.logger
/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br></br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
class CharacterIterator : DataSetIterator {
    //Valid characters
    private var validCharacters: CharArray

    //Maps each character to an index ind the input/output
    private var charToIdxMap: MutableMap<Char, Int>

    //All characters of the input file (after filtering to only those that are valid
    private var fileCharacters: CharArray

    //Length of each example/minibatch (number of characters)
    private var exampleLength: Int

    //Size of each minibatch (number of examples)
    private var miniBatchSize: Int
    private var rng: Random

    //Offsets for the start of each example
    private val exampleStartOffsets = LinkedList<Int>()

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    constructor(
        textFilePath: String, textFileEncoding: Charset?, miniBatchSize: Int, exampleLength: Int,
        validCharacters: CharArray, rng: Random
    ) {
        if (!File(textFilePath).exists()) throw IOException("Could not access file (does not exist): $textFilePath")
        require(miniBatchSize > 0) { "Invalid miniBatchSize (must be >0)" }
        this.validCharacters = validCharacters
        this.exampleLength = exampleLength
        this.miniBatchSize = miniBatchSize
        this.rng = rng

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = HashMap()
        for (i in validCharacters.indices) charToIdxMap[validCharacters[i]] = i

        //Load file and convert contents to a char[]
        val newLineValid = charToIdxMap.containsKey('\n')
        val lines = Files.readAllLines(File(textFilePath).toPath(), textFileEncoding)
        var maxSize = lines.size //add lines.size() to account for newline characters at end of each line
        for (s in lines) maxSize += s.length
        val characters = CharArray(maxSize)
        var currIdx = 0
        for (s in lines) {
            val thisLine = s.toCharArray()
            for (aThisLine in thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) continue
                characters[currIdx++] = aThisLine
            }
            if (newLineValid) characters[currIdx++] = '\n'
        }
        fileCharacters = if (currIdx == characters.size) {
            characters
        } else {
            Arrays.copyOfRange(characters, 0, currIdx)
        }
        require(exampleLength < fileCharacters.size) {
            ("exampleLength=" + exampleLength
                    + " cannot exceed number of valid characters in file (" + fileCharacters.size + ")")
        }
        val nRemoved = maxSize - fileCharacters.size
        logger.info(
            "Loaded and converted file: " + fileCharacters.size + " valid characters of "
                    + maxSize + " total characters (" + nRemoved + " removed)"
        )
        initializeOffsets()
    }

    constructor(
        lines: List<String>, miniBatchSize: Int, exampleLength: Int,
        validCharacters: CharArray, rng: Random
    ) {
        require(miniBatchSize > 0) { "Invalid miniBatchSize (must be >0)" }
        this.validCharacters = validCharacters
        this.exampleLength = exampleLength
        this.miniBatchSize = miniBatchSize
        this.rng = rng

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = HashMap()
        for (i in validCharacters.indices) charToIdxMap[validCharacters[i]] = i

        //Load file and convert contents to a char[]
        val newLineValid = charToIdxMap.containsKey('\n')
        var maxSize = lines.size //add lines.size() to account for newline characters at end of each line
        for (s in lines) maxSize += s.length
        val characters = CharArray(maxSize)
        var currIdx = 0
        for (s in lines) {
            val thisLine = s.toCharArray()
            for (aThisLine in thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) continue
                characters[currIdx++] = aThisLine
            }
            if (newLineValid) characters[currIdx++] = '\n'
        }
        fileCharacters = if (currIdx == characters.size) {
            characters
        } else {
            characters.copyOfRange(0, currIdx)
        }
        require(exampleLength < fileCharacters.size) {
            ("exampleLength=" + exampleLength
                    + " cannot exceed number of valid characters in file (" + fileCharacters.size + ")")
        }
        val nRemoved = maxSize - fileCharacters.size
        logger.info(
            "Loaded and converted file: " + fileCharacters.size + " valid characters of "
                    + maxSize + " total characters (" + nRemoved + " removed)"
        )
        initializeOffsets()
    }

    fun convertIndexToCharacter(idx: Int): Char {
        return validCharacters[idx]
    }

    fun convertCharacterToIndex(c: Char): Int {
        return charToIdxMap[c]!!
    }

    val randomCharacter: Char
        get() = validCharacters[(rng.nextDouble() * validCharacters.size).toInt()]

    override fun hasNext(): Boolean {
        return exampleStartOffsets.size > 0
    }

    override fun next(): DataSet {
        return next(miniBatchSize)
    }

    override fun next(num: Int): DataSet {
        if (exampleStartOffsets.size == 0) throw NoSuchElementException()
        val currMinibatchSize = Math.min(num, exampleStartOffsets.size)
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        val input = Nd4j.create(intArrayOf(currMinibatchSize, validCharacters.size, exampleLength), 'f')
        val labels = Nd4j.create(intArrayOf(currMinibatchSize, validCharacters.size, exampleLength), 'f')
        for (i in 0 until currMinibatchSize) {
            val startIdx = exampleStartOffsets.removeFirst()
            val endIdx = startIdx + exampleLength
            var currCharIdx = charToIdxMap[fileCharacters[startIdx]]!! //Current input
            var c = 0
            var j = startIdx + 1
            while (j < endIdx) {
                val nextCharIdx = charToIdxMap[fileCharacters[j]]!! //Next character to predict
                input.putScalar(intArrayOf(i, currCharIdx, c), 1.0)
                labels.putScalar(intArrayOf(i, nextCharIdx, c), 1.0)
                currCharIdx = nextCharIdx
                j++
                c++
            }
        }
        return DataSet(input, labels)
    }

    fun totalExamples(): Int {
        return (fileCharacters.size - 1) / miniBatchSize - 2
    }

    override fun inputColumns(): Int {
        return validCharacters.size
    }

    override fun totalOutcomes(): Int {
        return validCharacters.size
    }

    override fun reset() {
        exampleStartOffsets.clear()
        initializeOffsets()
    }

    private fun initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        val nMinibatchesPerEpoch =
            (fileCharacters.size - 1) / exampleLength - 2 //-2: for end index, and for partial example
        for (i in 0 until nMinibatchesPerEpoch) {
            exampleStartOffsets.add(i * exampleLength)
        }
        exampleStartOffsets.shuffle(rng)
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        return true
    }

    override fun batch(): Int {
        return miniBatchSize
    }

    fun cursor(): Int {
        return totalExamples() - exampleStartOffsets.size
    }

    fun numExamples(): Int {
        return totalExamples()
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun getLabels(): List<String> {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    companion object {//	for(char c='A'; c<='Z'; c++) validChars.add(c);
        /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc  */
        val minimalCharacterSet: CharArray
            get() {
                val validChars: MutableList<Char> = LinkedList()
                run {
                    var c = '가'
                    while (c <= '힣') {
                        validChars.add(c)
                        c++
                    }
                }
                //	for(char c='A'; c<='Z'; c++) validChars.add(c);
                var c = '0'
                while (c <= '9') {
                    validChars.add(c)
                    c++
                }
                val temp = charArrayOf('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
                for (c in temp) validChars.add(c)
                val out = CharArray(validChars.size)
                var i = 0
                for (c in validChars) out[i++] = c
                return out
            }

        /** As per getMinimalCharacterSet(), but with a few extra characters  */
        val defaultCharacterSet: CharArray
            get() {
                val validChars: MutableList<Char> = LinkedList()
                for (c in minimalCharacterSet) validChars.add(c)
                val additionalChars = charArrayOf(
                    '@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
                    '\\', '|', '<', '>'
                )
                for (c in additionalChars) validChars.add(c)
                val out = CharArray(validChars.size)
                var i = 0
                for (c in validChars) out[i++] = c
                return out
            }
    }
}