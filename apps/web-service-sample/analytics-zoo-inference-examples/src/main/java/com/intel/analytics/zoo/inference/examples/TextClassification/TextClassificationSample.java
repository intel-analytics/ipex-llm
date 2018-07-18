package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import java.util.ArrayList;
import java.util.List;

public class TextClassificationSample {
    public void main(String[] args) {
        String baseDir = System.getProperty("baseDir", "/home/yidiyang/workspace");
        String modelPath = System.getProperty("modelPath", baseDir + "/model/textClassification/textClassificationModel");
        int stopWordsCount = 1;
        int sequenceLength = 500;
        TextClassificationModel model = new TextClassificationModel(stopWordsCount, sequenceLength);
        long begin = System.currentTimeMillis();
        String sampleText = "it is for test";
        JTensor input = model.preProcess(sampleText);
        long end = System.currentTimeMillis();
        long processTime = end - begin;
        model.load(modelPath);
        begin = System.currentTimeMillis();
        List<JTensor> inputList = new ArrayList<>();
        inputList.add(input);
        List<List<JTensor>> result = model.predict(inputList);
        end = System.currentTimeMillis();
        long predictTime = end - begin;
        JTensor resultTensor = result.get(0).get(0);
        List<Float> resultDis = resultTensor.getData();
        int resultClass = 0;
        float maxProb = 0;
        for (int i = 0; i < resultDis.size(); i++) {
            if (resultDis.get(i) >= maxProb) {
                resultClass = i;
                maxProb = resultDis.get(i);
            }
        }
        String answer = String.format("The predict class is:%s\nThe probability distribution is:%s \n#Process Time elapsed : %d, Predict Time elapsed: %d", Integer.toString(resultClass), resultDis.toString(), processTime, predictTime);
        System.out.println(answer);
    }
}