package com.intel.analytics.zoo.apps.textclassfication.inference;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class SimpleDriver {

    public static void main(String[] args) {
        String embeddingFilePath = System.getProperty("EMBEDDING_FILE_PATH", "./glove.6B.300d.txt");
        String modelPath = System.getProperty("MODEL_PATH", "./models/text-classification.bigdl");
        TextClassificationModel textClassificationModel = new TextClassificationModel(10, 10, 200, embeddingFilePath);
        textClassificationModel.load(modelPath);
        String[] texts = new String[]{
                "Sorry, Gregg, it was no answer to a post of mine. And you are quite\n" +
                        "fond of using abusing language whenever you think your religion is\n" +
                        "misrepresented. By the way, I have no trouble telling me apart from\n" +
                        "Bob Beauchaine.\n" +
                        " \n" +
                        " \n" +
                        "I still wait for your answer to that the Quran allows you to beat your wife\n" +
                        "into submission. You were quite upset about the claim that it was in it,\n" +
                        "to be more correct, you said it wasn't.\n" +
                        " \n" +
                        "I asked you about what your consequences were in case it would be in the\n" +
                        "Quran, but you have simply ceased to respond on that thread. Can it be\n" +
                        "that you have found out in the meantime that it is the Holy Book?\n" +
                        " \n" +
                        "What are your consequences now? Was your being upset just a show? Do you\n" +
                        "simple inherit your morals from a Book, ie is it suddenly ok now? Is it\n" +
                        "correct to say that the words of Muhammad reflect the primitive Machism\n" +
                        "of his society? Or have you spent your time with your new gained freedom?\n" +
                        "   Benedikt",
                "From: pharvey@quack.kfu.com (Paul Harvey)\n" +
                        "Subject: Re: I'll see your demand and raise you... (was Re: After 2000 years etc)\n" +
                        "\n" +
                        "In article <C64H4w.BFH@darkside.osrhe.uoknor.edu> \n" +
                        "bil@okcforum.osrhe.edu (Bill Conner) writes:\n" +
                        ">Keith M. Ryan (kmr4@po.CWRU.edu) wrote:\n" +
                        ">: [34mAnd now . . . [35mDeep Thoughts[0m\n" +
                        ">: \t[32mby Jack Handey.[0m\n" +
                        ">: [36mIf you go parachuting, and your parachute doesn't open, and your\n" +
                        ">: friends are all watching you fall, I think a funny gag would be\n" +
                        ">: to pretend you were swimming.[0m\n" +
                        ">Keith, \n" +
                        ">As you must know by now there are no Escape Sequences here (ANSI or\n" +
                        ">otherwise). Once you enter here, your terminal beomes dumb. There's\n" +
                        ">something significant about all this ...\n" +
                        "\n" +
                        "You are in the village. Many happy returns! Be seeing you!\n" +
                        "\n" +
                        "[your ways and means get reign of the tek!]"};
        List<List<JTensor>> inputs = new ArrayList<List<JTensor>>();
        for (String text : texts) {
            List<JTensor> input = new ArrayList<JTensor>();
            JTensor inputTensor = textClassificationModel.preprocess(text);
            input.add(inputTensor);
            System.out.println(inputTensor);
            inputs.add(input);
        }
        List<List<JTensor>> results = textClassificationModel.predict(inputs);
        System.out.println(results);
        for(List<JTensor> result : results) {
            float[] data = result.get(0).getData();
            float max = 0 ;
            int classed = 0;
            for(int i =0; i< data.length; i++){
                if(data[i] > max) {
                    max = data[i];
                    classed = i;
                }
            }
            System.out.println("class " + classed);
        }

    }
}
