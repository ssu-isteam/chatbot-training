package dev.isteam.chatbot.dl.api.dataset.loader




internal class MINDsLabDataSetLoaderTest {


    fun load() {
        /**
        var path = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\일반상식\\02_squad_질문_답변_제시문_말뭉치\\ko_wiki_v1_squad.json"
        var path2 = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\기계독해\\기계독해분야\\ko_nia_normal_squad_all.json"
        var loader = MINDsLabDataSetLoader(path)
        var loader2 = MINDsLabDataSetLoader(path2)
        var packedRawDataSet = loader.load().get()
        var packedRawDataSet2 = loader2.load().get()

        packedRawDataSet.rawDataSets.plus(packedRawDataSet2.rawDataSets)

        var tokenizerFactory = KoreanTokenizerFactory()
        tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

        var iterator = RawDataSetIterator(packedRawDataSet,RawDataSetIterator.IterativeType.QUESTION)
        val cache = InMemoryLookupCache()
        val table: WeightLookupTable<VocabWord> = InMemoryLookupTable.Builder<VocabWord>()
            .vectorLength(100)
            .useAdaGrad(false)
            .cache(cache)
            .lr(0.025).build()

        val vec: Word2Vec = Word2Vec.Builder()
            .minWordFrequency(5)
            .iterations(1)
            .epochs(1)
            .layerSize(100)
            .batchSize(100000)
            .seed(42)
            .windowSize(5)
            .iterate(iterator)
            .tokenizerFactory(tokenizerFactory)
            .lookupTable(table)
            .vocabCache(cache)
            .build()
        vec.fit()
        WordVectorSerializer.writeWord2VecModel(vec,"word2Vec.w2v")
        **/
    /**
        var path = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\일반상식\\02_squad_질문_답변_제시문_말뭉치\\ko_wiki_v1_squad.json"
        var path2 = "C:\\Users\\Singlerr\\Desktop\\ISTEAM\\AI 데이터 셋\\기계독해\\기계독해분야\\ko_nia_normal_squad_all.json"
        var loader = MINDsLabDataSetLoader(path)
        var loader2 = MINDsLabDataSetLoader(path2)
        var packedRawDataSet = loader.load().get()
        var packedRawDataSet2 = loader2.load().get()

        packedRawDataSet.rawDataSets.plus(packedRawDataSet2.rawDataSets)


        var tokenizerFactory = KoreanTokenizerFactory()
        tokenizerFactory.tokenPreProcessor = KoreanTokenPreprocessor()

        var vec = WordVectorSerializer.readWord2VecModel("word2VecModel.w2v")


        var batchSize = 10

        var dataSource = DataSource(packedRawDataSet = packedRawDataSet, word2Vec = vec, koreanTokenizerFactory = tokenizerFactory, batchSize = batchSize)

        var network = KoreanNeuralNetwork.buildNeuralNetwork(100,packedRawDataSet.rawDataSets.size)
        network.init()


        var k = 0
        println("작업 시작")
        for(i in 0 until 1){
            while(dataSource.hasNext()){
                var dataSet = dataSource.next()
                network.fit(dataSet)
                k += batchSize
                println("현재 epoch: ${i}, 현재 epoch 진행도: ${k}/${dataSource.totalOutcomes()}")
            }
            k = 0
            println("다시 시작")
            dataSource.reset()
        }
        ModelSerializer.writeModel(network,"network.model",true)
        **/
    }
}