package dev.isteam.chatbot.dl.api.dataset.preprocessor

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess

class KoreanTokenPreprocessor : TokenPreProcess {
    /**
     * Pre process a token
     * @param token the token to pre process
     * @return the preprocessed token
     */
    override fun preProcess(token: String?): String {
        return token!!.replace("'\\([^)]*\\)", "")
    }

}