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

import com.intel.analytics.bigdl.utils.Engine;
import scala.actors.threadpool.Arrays;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

public abstract class AbstractInferenceModel implements Serializable {

  private int supportedConcurrentNum = 1;
  private FloatInferenceModel originalModel;
  protected LinkedBlockingQueue<FloatInferenceModel> modelQueue;

  public AbstractInferenceModel() {
    modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
  }

  public AbstractInferenceModel(int supportedConcurrentNum) {
    this.supportedConcurrentNum = supportedConcurrentNum;
    modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
  }

  public void load(String modelPath) {
    load(modelPath, null);
  }

  public void load(String modelPath, String weightPath) {
    originalModel = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath);
    FloatInferenceModel[] modelArray = InferenceModelFactory.cloneSharedWeightsModelsIntoArray(originalModel, supportedConcurrentNum);
    for (int i = 0; i < supportedConcurrentNum; i++) {
      modelQueue.offer(modelArray[i]);
    }
  }

  public void loadCaffe(String modelPath) {
    loadCaffe(modelPath, null);
  }

  public void loadCaffe(String modelPath, String weightPath) {
    originalModel = InferenceModelFactory.loadFloatInferenceModelForCaffe(modelPath, weightPath);
    FloatInferenceModel[] modelArray = InferenceModelFactory.cloneSharedWeightsModelsIntoArray(originalModel, supportedConcurrentNum);
    for (int i = 0; i < supportedConcurrentNum; i++) {
      modelQueue.offer(modelArray[i]);
    }
  }

  public void loadTF(String modelPath) {
    loadTF(modelPath, 1, 1, true);
  }

  public void loadTF(String modelPath, int intraOpParallelismThreads, int interOpParallelismThreads, boolean usePerSessionThreads) {
    originalModel = InferenceModelFactory.loadFloatInferenceModelForTF(modelPath, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads);
    FloatInferenceModel[] modelArray = InferenceModelFactory.cloneSharedWeightsModelsIntoArray(originalModel, supportedConcurrentNum);
    for (int i = 0; i < supportedConcurrentNum; i++) {
      modelQueue.offer(modelArray[i]);
    }
  }

  public void reload(String modelPath) {
    load(modelPath, null);
  }

  public void reload(String modelPath, String weightPath) {
    originalModel = null;
    modelQueue.clear();
    load(modelPath, weightPath);
  }

  @Deprecated
  public List<Float> predict(List<Float> input, int... shape) {
    FloatInferenceModel model = null;
    List<Float> result;
    List<Integer> inputShape = new ArrayList<Integer>();
    for (int s : shape) {
      inputShape.add(s);
    }
    try {
      model = modelQueue.take();
    } catch (InterruptedException e) {
      throw new InferenceRuntimeException("no model available", e);
    }
    try {
      result = model.predict(input, inputShape);
    } finally {
      modelQueue.offer(model);
    }
    return result;
  }

  public List<List<JTensor>> predict(List<List<JTensor>> inputs) {
    FloatInferenceModel model = null;
    List<List<JTensor>> result;
    try {
      model = modelQueue.take();
    } catch (InterruptedException e) {
      throw new InferenceRuntimeException("no model available", e);
    }
    try {
      result = model.predict(inputs);
    } finally {
      modelQueue.offer(model);
    }
    return result;
  }

  public List<List<JTensor>> predict(List<JTensor>[] inputs) {
    return predict(Arrays.asList(inputs));
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    out.writeInt(supportedConcurrentNum);
    out.writeObject(originalModel);
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    System.setProperty("bigdl.localMode", System.getProperty("bigdl.localMode", "true"));
    System.setProperty("bigdl.coreNumber", System.getProperty("bigdl.coreNumber", "1"));
    Engine.init();
    supportedConcurrentNum = in.readInt();
    originalModel = ((FloatInferenceModel) in.readObject());
    modelQueue = new LinkedBlockingQueue<FloatInferenceModel>(supportedConcurrentNum);
    FloatInferenceModel[] modelArray = InferenceModelFactory.cloneSharedWeightsModelsIntoArray(originalModel, supportedConcurrentNum);
    for (int i = 0; i < supportedConcurrentNum; i++) {
      modelQueue.offer(modelArray[i]);
    }
  }
}