package dev.isteam.chatbot.dl.api.dataset.loader

import dev.isteam.chatbot.dl.api.dataset.DataSetLoader
import dev.isteam.chatbot.dl.api.dataset.LSTMPackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import org.json.JSONObject
import java.io.FileInputStream
import java.util.concurrent.CompletableFuture

class VIVEDataSetLoader(private val paths: Array<String>) : DataSetLoader {
    @Deprecated("VIVEDataSetLoader does not support load() : PackedRawDataSet, use loadDialogues() : List<LSTMPackedRawDataSet>")
    override fun load(): CompletableFuture<PackedRawDataSet> {
        return CompletableFuture.supplyAsync {
            return@supplyAsync null
        }
    }

    fun loadDialogues(): CompletableFuture<List<LSTMPackedRawDataSet>> {
        return CompletableFuture.supplyAsync {
            var dataSets = arrayListOf<LSTMPackedRawDataSet>()

            for (path in paths) {
                var (dialogues, max) = load(path)
                dataSets.add(LSTMPackedRawDataSet(max, dialogues))
            }
            return@supplyAsync dataSets
        }
    }

    private fun load(path: String): Pair<MutableList<PackedRawDataSet>, Int> {
        var max = 0

        var dataSets = arrayListOf<PackedRawDataSet>()
        val rawText = FileInputStream(path).reader(charset = Charsets.UTF_8).readText()
        var obj = JSONObject(rawText)
        var dataArray = obj.getJSONArray("data")



        for (i in 0 until dataArray.length()) {
            var packedRawDataSet = PackedRawDataSet()
            var ctx = dataArray.getJSONObject(i)
            var body = ctx.getJSONObject("body")
            var dialogues = body.getJSONArray("dialogue")

            if (dialogues.length() > max)
                max = dialogues.length()

            for (j in 0 until dialogues.length()) {
                var dialogue = dialogues.getJSONObject(j)
                var utterance = dialogue.getString("utterance")
                var dataSet = RawDataSet(question = utterance, answer = utterance)
                packedRawDataSet.rawDataSets.add(dataSet)

            }
            dataSets.add(packedRawDataSet)
        }
        return dataSets to max
    }

}