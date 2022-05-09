package dev.isteam.chatbot.dl.api.dataset

class PackedRawDataSet(val rawDataSets: MutableList<RawDataSet> = ArrayList()){
    fun size() = rawDataSets.size
    operator fun get(i:Int) : String{
        return rawDataSets[i].question!!
    }
}