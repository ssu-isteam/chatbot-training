package dev.isteam.chatbot.dl.api.tokenizer

import kr.co.shineware.nlp.komoran.model.Token
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer

class KoreanTokenizer(private val tokenList: List<Token>, var iterator: Iterator<Token> = tokenList.iterator()) :
    Tokenizer {

    private var tokenPreProcess: TokenPreProcess? = null

    /**
     * An iterator for tracking whether
     * more tokens are left in the iterator not
     * @return whether there is anymore tokens
     * to iterate over
     */
    override fun hasMoreTokens(): Boolean {
        return iterator.hasNext()
    }

    /**
     * The number of tokens in the tokenizer
     * @return the number of tokens
     */
    override fun countTokens(): Int {
        return tokenList.size
    }

    /**
     * The next token (word usually) in the string
     * @return the next token in the string if any
     */
    override fun nextToken(): String {
        return if (tokenPreProcess != null) tokenPreProcess!!.preProcess(iterator.next().morph) else iterator.next().morph
    }

    /**
     * Returns a list of all the tokens
     * @return a list of all the tokens
     */
    override fun getTokens(): MutableList<String> {
        return if (tokenPreProcess != null) tokenList.map { token ->
            tokenPreProcess!!.preProcess(token.morph)
        }.toMutableList() else tokenList.map { token ->
            tokenPreProcess!!.preProcess(token.morph)
        }.toMutableList()
    }

    /**
     * Set the token pre process
     * @param tokenPreProcessor the token pre processor to set
     */
    override fun setTokenPreProcessor(tokenPreProcessor: TokenPreProcess?) {
        tokenPreProcess = tokenPreProcessor
    }

}