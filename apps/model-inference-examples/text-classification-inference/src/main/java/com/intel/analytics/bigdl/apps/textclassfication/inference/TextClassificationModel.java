package com.intel.analytics.bigdl.apps.textclassfication.inference;

import com.intel.analytics.bigdl.serving.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.bigdl.serving.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    private int stopWordsCount;
    private int sequenceLength;
    private String embeddingFilePath;
    private TextProcessor textProcessor;

    public TextClassificationModel(int supportedConcurrentNum, int stopWordsCount, int sequenceLength, String embeddingFilePath) {
        super(supportedConcurrentNum);
        this.stopWordsCount = stopWordsCount;
        this.sequenceLength = sequenceLength;
        this.embeddingFilePath = embeddingFilePath;
        this.textProcessor = new TextProcessor(this.stopWordsCount, this.sequenceLength, this.embeddingFilePath);
    }

    public JTensor preprocess(String text) {
        return this.textProcessor.preprocess(text);
    }
}
