package dev.isteam.chatbot.dl.api.dataset

data class LSTMPackedRawDataSet(val max:Int, val dialogues:MutableList<PackedRawDataSet>) : PackedRawDataSet(dialogues.map { it.rawDataSets }.flatten().toMutableList())