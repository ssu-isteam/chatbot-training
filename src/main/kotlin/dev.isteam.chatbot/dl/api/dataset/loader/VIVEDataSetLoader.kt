package dev.isteam.chatbot.dl.api.dataset.loader

import dev.isteam.chatbot.dl.api.dataset.DataSetLoader
import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import me.tongfei.progressbar.ProgressBar
import org.json.JSONObject
import java.io.FileInputStream
import java.util.concurrent.CompletableFuture

class VIVEDataSetLoader(private val paths:Array<String>) : DataSetLoader{
    override fun load(): CompletableFuture<PackedRawDataSet> {
        return CompletableFuture.supplyAsync{
            var pb = ProgressBar("Reading files", paths.size.toLong())

            var packedRawDataSet = PackedRawDataSet()
            for (path in paths) {
                var dataSets = load(path)
                pb.step()
                packedRawDataSet.rawDataSets.addAll(dataSets)
            }
            return@supplyAsync packedRawDataSet
        }
    }

    private fun load(path:String) : List<RawDataSet>{
        var dataSets = arrayListOf<RawDataSet>()

        val rawText = FileInputStream(path).reader(charset = Charsets.UTF_8).readText()

        var obj = JSONObject(rawText)

        var dataArray = obj.getJSONArray("data")

        for(i in 0 until dataArray.length()){
            var ctx = dataArray.getJSONObject(i)
            var body = ctx.getJSONObject("body")
            var dialogues = body.getJSONArray("dialogue")

            for(j in 0 until dialogues.length()){
                var dialogue = dialogues.getJSONObject(j)
                var utterance = dialogue.getString("utterance")
                var dataSet = RawDataSet(question = utterance, answer =utterance)
                dataSets.add(dataSet)

            }
        }
        return dataSets
    }

}