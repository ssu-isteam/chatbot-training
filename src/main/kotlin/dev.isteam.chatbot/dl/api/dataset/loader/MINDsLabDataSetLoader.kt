package dev.isteam.chatbot.dl.api.dataset.loader

import dev.isteam.chatbot.dl.api.dataset.DataSetLoader
import dev.isteam.chatbot.dl.api.dataset.PackedRawDataSet
import dev.isteam.chatbot.dl.api.dataset.RawDataSet
import org.json.JSONObject
import java.io.FileInputStream
import java.io.IOException
import java.util.concurrent.CompletableFuture

class MINDsLabDataSetLoader(private val path: String) : DataSetLoader {
    override fun load(): CompletableFuture<PackedRawDataSet> {
        return CompletableFuture.supplyAsync {
            try {
                var text = FileInputStream(path).reader(charset = Charsets.UTF_8).readText()
                var obj = JSONObject(text)

                var data = obj.getJSONArray("data")


                var datasets = ArrayList<RawDataSet>()

                for (i in 0 until data.length()) {
                    var dataElem = data.getJSONObject(i)
                    var paragraphs = dataElem.getJSONArray("paragraphs")
                    for (j in 0 until paragraphs.length()) {
                        var paragraph = paragraphs.getJSONObject(j)
                        var qas = paragraph.getJSONArray("qas")

                        for (k in 0 until qas.length()) {
                            var qasElem = qas.getJSONObject(k)

                            var question = qasElem.getString("question")

                            var answers = qasElem.getJSONArray("answers")
                            for (l in 0 until answers.length()) {
                                var answerElem = answers.getJSONObject(l)
                                var answer = answerElem.getString("text")
                                datasets.add(RawDataSet(question = question, answer = answer))
                            }
                        }
                    }
                }

                return@supplyAsync PackedRawDataSet(datasets)
            } catch (ex: IOException) {

            }
            return@supplyAsync PackedRawDataSet()
        }
    }


}