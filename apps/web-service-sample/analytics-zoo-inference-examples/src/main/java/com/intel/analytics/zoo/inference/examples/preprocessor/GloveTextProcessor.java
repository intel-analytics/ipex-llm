package com.intel.analytics.zoo.inference.examples.preprocessor;

import com.intel.analytics.zoo.preprocess.ITextProcessing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class GloveTextProcessor extends ITextProcessing {
    public Map<String, List<Float>> embMap;

    public GloveTextProcessor(String embFilePath){
        embMap = loadEmbedding(embFilePath);
    }

    @Override
    public Map<String, List<Float>> loadEmbedding(String embFilePath) {
        Map<String, List<Float>> embMap = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(embFilePath))) {
            for (String line; (line = br.readLine()) != null; ) {
                String[] parts = line.split(" ", 2);
                String word = parts[0];
                String emb = parts[1];
                Scanner scanner = new Scanner(emb);
                List<Float> list = new ArrayList<>();
                while (scanner.hasNextFloat()) {
                    list.add(scanner.nextFloat());
                }
                embMap.put(word, list);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return embMap;
    }

    // @Override
//	public JTensor preprocess(String text) {
//		List<String> tokens = tokenize(text);
//		List<String> shapedTokens = shaping(stopWords(tokens,1),500);
//		Map<String, List<Float>> embMap = loadEmbedding(System.getProperty("EMBEDDING_PATH", "/home/yidiyang/workspace/dataset/glove.6B"));
//		List<List<Float>> vectorizedTokens = vectorize(shapedTokens, embMap);
//		List<Float> data = Lists.newArrayList(Iterables.concat(vectorizedTokens));
//		List<Integer> shape = new ArrayList<>();
//		shape.add(vectorizedTokens.size());
//		shape.add(vectorizedTokens.get(0).size());
//		JTensor tensorInput = new JTensor(data, shape);
//		return tensorInput;
//	}
}



