package dev.isteam.chatbot.dl.api.dataset

import java.util.concurrent.CompletableFuture


interface DataSetLoader {
    fun load(): CompletableFuture<PackedRawDataSet>

}