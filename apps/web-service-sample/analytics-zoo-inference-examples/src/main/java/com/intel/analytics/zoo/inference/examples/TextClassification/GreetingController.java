package com.intel.analytics.zoo.inference.examples.TextClassification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

@RestController
public class GreetingController {

    private static final String template = "Hello, %s!";
    private final AtomicLong counter = new AtomicLong();
    private String current = System.getProperty("user.dir");
    private String modelPath = System.getProperty("modelPath", current + "/src/main/resources/textClassificationModel");
    private int stopWordsCount = 1;
    private int sequenceLength = 500;
    private TextClassificationModel model;
    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public GreetingController() {
        this.model = new TextClassificationModel(this.stopWordsCount, this.sequenceLength);
        //default to use the model in resuources
        this.model.load(this.modelPath);
    }

    @RequestMapping(value = "/greeting")
    public Greeting greeting(@RequestParam(value = "name", defaultValue = "World") String name) {
        return new Greeting(counter.incrementAndGet(),
                String.format(template, name));
    }

    @RequestMapping(value = "/test", method = {RequestMethod.POST})
    public Greeting testing(@RequestParam(value = "name", defaultValue = "test") String name) {
        return new Greeting(counter.incrementAndGet(),
                String.format(template, name));
    }

    @RequestMapping(value = "/predict", method = {RequestMethod.POST})
    public String webPredict(@RequestBody String text) {
        if (!text.isEmpty()) {
            long begin = System.currentTimeMillis();
            JTensor input = model.preProcess(text);
            long end = System.currentTimeMillis();
            long processTime = end - begin;
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
            String time = String.format("Process Time elapsed : %d, Predict Time elapsed: %d", processTime, predictTime);
            logger.info(time);
            String answer = String.format("The predict class is:%s\nThe probability distribution is:%s \n", Integer.toString(resultClass), resultDis.toString());
            return answer;
        } else {
            return "error,no text found";
        }

    }
}
