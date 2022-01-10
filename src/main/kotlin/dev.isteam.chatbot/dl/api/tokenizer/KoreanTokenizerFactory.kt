package dev.isteam.chatbot.dl.api.tokenizer

import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL
import kr.co.shineware.nlp.komoran.core.Komoran
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import java.io.InputStream
import java.nio.charset.Charset

class KoreanTokenizerFactory(private val koreanAnalyzer: Komoran = Komoran(DEFAULT_MODEL.FULL)) : TokenizerFactory {

    private var tokenPreProcess: TokenPreProcess? = null

    /**
     * The tokenizer to createComplex
     * @param toTokenize the string to createComplex the tokenizer with
     * @return the new tokenizer
     */
    override fun create(toTokenize: String?): Tokenizer {
        var result = koreanAnalyzer.analyze(toTokenize)
        var tokenizer = KoreanTokenizer(result.tokenList)
        if (tokenPreProcess != null)
            tokenizer.setTokenPreProcessor(tokenPreProcessor)
        return tokenizer
    }

    /**
     * Create a tokenizer based on an input stream
     * @param toTokenize
     * @return
     */
    override fun create(toTokenize: InputStream?): Tokenizer {
        var text = toTokenize!!.reader(Charset.forName("UTF-8")).readText()
        var result = koreanAnalyzer.analyze(text)
        var tokenizer = KoreanTokenizer(result.tokenList)
        if (tokenPreProcess != null)
            tokenizer.setTokenPreProcessor(tokenPreProcessor)
        return tokenizer
    }

    /**
     * Sets a token pre processor to be used
     * with every tokenizer
     * @param preProcessor the token pre processor to use
     */
    override fun setTokenPreProcessor(preProcessor: TokenPreProcess?) {
        tokenPreProcess = preProcessor!!
    }

    /**
     * Returns TokenPreProcessor set for this TokenizerFactory instance
     *
     * @return TokenPreProcessor instance, or null if no preprocessor was defined
     */
    override fun getTokenPreProcessor(): TokenPreProcess {
        return tokenPreProcess!!
    }

}