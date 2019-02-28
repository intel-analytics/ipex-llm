/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.inference;

import scala.actors.threadpool.Arrays;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractInferenceModel extends InferenceModel implements Serializable {

  public AbstractInferenceModel() {
    super(1, null, null);
  }

  public AbstractInferenceModel(int supportedConcurrentNum) {
    super(supportedConcurrentNum, null, null);
  }

  public void load(String modelPath) {
    doLoad(modelPath, null);
  }

  public void load(String modelPath, String weightPath) {
    doLoad(modelPath, weightPath);
  }

  public void loadCaffe(String modelPath) {
    doLoadCaffe(modelPath, null);
  }

  public void loadCaffe(String modelPath, String weightPath) {
    doLoadCaffe(modelPath, weightPath);
  }

  public void loadTF(String modelPath) {
    doLoadTF(modelPath);
  }

  public void loadTF(String modelPath, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    doLoadTF(modelPath, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void loadTF(String modelPath, String modelType) {
    doLoadTF(modelPath, modelType);
  }

  public void loadTF(String modelPath, String pipelineConfigFilePath, String extensionsConfigFilePath) {
    doLoadTF(modelPath, pipelineConfigFilePath, extensionsConfigFilePath);
  }

  public void loadTF(String modelPath, String modelType, String pipelineConfigFilePath, String extensionsConfigFilePath) {
    doLoadTF(modelPath, modelType, pipelineConfigFilePath, extensionsConfigFilePath);
  }

  public void loadOpenVINO(String modelFilePath, String weightFilePath) {
    doLoadOpenVINO(modelFilePath, weightFilePath);
  }

  public void reload(String modelPath) {
    doReload(modelPath, null);
  }

  public void reload(String modelPath, String weightPath) {
    doReload(modelPath, weightPath);
  }

  @Deprecated
  public List<Float> predict(List<Float> input, int... shape) {
    List<Integer> inputShape = new ArrayList<Integer>();
    for (int s : shape) {
      inputShape.add(s);
    }
    return doPredict(input, inputShape);
  }

  public List<List<JTensor>> predict(List<List<JTensor>> inputs) {
    return doPredict(inputs);
  }

  public List<List<JTensor>> predict(List<JTensor>[] inputs) {
    return predict(Arrays.asList(inputs));
  }

  @Override
  public String toString() {
    return super.toString();
  }
}