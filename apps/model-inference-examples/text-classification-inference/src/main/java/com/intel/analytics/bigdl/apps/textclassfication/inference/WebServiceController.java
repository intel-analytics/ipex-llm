package com.intel.analytics.bigdl.apps.textclassfication.inference;

import com.intel.analytics.bigdl.orca.inference.JTensor;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
public class WebServiceController {

    private TextClassificationModel textClassificationModel;

    public WebServiceController() {
        String embeddingFilePath = System.getProperty("EMBEDDING_FILE_PATH", "./glove.6B.300d.txt");
        String modelPath = System.getProperty("MODEL_PATH", "./models/text-classification.bigdl");
        textClassificationModel = new TextClassificationModel(10, 10, 200, embeddingFilePath);
        textClassificationModel.load(modelPath);
    }

    @RequestMapping(value = "/")
    public String greeting() {
        return "welcome!";
    }

    @RequestMapping(value = "/greetings")
    public String greeting(@RequestParam(value = "name", defaultValue = "World") String name) {
        return "hello, " + name;
    }

    @RequestMapping(value = "/predict", method = {RequestMethod.POST})
    public String webPredict(@RequestBody String text) {
        if (!text.isEmpty()) {
            List<List<JTensor>> inputs = new ArrayList<List<JTensor>>();
            List<JTensor> input = new ArrayList<JTensor>();
            JTensor inputTensor = textClassificationModel.preprocess(text);
            input.add(inputTensor);
            inputs.add(input);
            List<List<JTensor>> results = textClassificationModel.predict(inputs);
            float[] data = results.get(0).get(0).getData();
            float max = 0;
            int classed = 0;
            for (int i = 0; i < data.length; i++) {
                if (data[i] > max) {
                    max = data[i];
                    classed = i;
                }
            }
            System.out.println("class " + classed);
            String answer = String.format("The predict class is:%s\nThe probability is:%s \n", classed, max);
            return answer;
        } else {
            return "error,no text found";
        }
    }


}
