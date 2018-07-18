package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.inference.examples.preprocessor.GloveTextProcessor;
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    private int stopWordsCount, sequenceLength;

    public TextClassificationModel(int stopWordsCount, int sequenceLength) {
        this.stopWordsCount = stopWordsCount;
        this.sequenceLength = sequenceLength;
    }

    private GloveTextProcessor preprocessor = new GloveTextProcessor(System.getProperty("EMBEDDING_PATH", "/home/yidiyang/workspace/dataset/glove.6B/glove.6B.200d.txt"));

    public JTensor preProcess(String text) {
        JTensor input = preprocessor.preprocessWithEmbMap(text, stopWordsCount, sequenceLength, preprocessor.embMap);
        return input;
    }
}
