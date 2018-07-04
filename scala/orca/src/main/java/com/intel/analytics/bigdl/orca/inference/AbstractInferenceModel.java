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

import com.intel.analytics.zoo.pipeline.api.net.TFNet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractInferenceModel implements Serializable {
  private FloatInferenceModel model;
  private int supportedConcurrentNum = 1;

  public AbstractInferenceModel() {
  }

  public AbstractInferenceModel(int supportedConcurrentNum) {
    this.supportedConcurrentNum = supportedConcurrentNum;
  }

  public void load(String modelPath) {
    load(modelPath, null);
  }

  public void load(String modelPath, String weightPath) {
    this.model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
  }

  public void loadCaffe(String modelPath) {
    loadCaffe(modelPath, null);
  }

  public void loadCaffe(String modelPath, String weightPath) {
    this.model = InferenceModelFactory.loadFloatInferenceModelForCaffe(modelPath, weightPath);
  }

  public void loadTF(String modelPath) {
    this.model = InferenceModelFactory.loadFloatInferenceModelForTF(modelPath,
            1, 1, true);
  }

  public void loadTF(String modelPath,
                     int intraOpParallelismThreads,
                     int interOpParallelismThreads,
                     boolean usePerSessionThreads) {
    this.model = InferenceModelFactory.loadFloatInferenceModelForTF(modelPath,
            intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
  }

  public void reload(String modelPath) {
    load(modelPath, null);
  }

  public void reload(String modelPath, String weightPath) {
    this.model = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
  }

  @Deprecated
  public List<Float> predict(List<Float> input, int... shape) {
    List<Integer> inputShape = new ArrayList<Integer>();
    for (int s : shape) {
      inputShape.add(s);
    }
    return model.predict(input, inputShape);
  }

  public List<List<JTensor>> predict(List<JTensor> inputs) {
    return model.predict(inputs);
  }

}
