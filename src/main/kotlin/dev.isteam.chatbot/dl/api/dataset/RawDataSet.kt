package dev.isteam.chatbot.dl.api.dataset

class RawDataSet(var question: String?, val answer: String?)
data class Word2VecRawDataSet(val x: MutableList<String>, val y: MutableList<String>) {
    fun size() = x.size
}