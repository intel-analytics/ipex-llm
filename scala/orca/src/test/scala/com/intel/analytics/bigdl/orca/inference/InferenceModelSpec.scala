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

package com.intel.analytics.zoo.pipeline.inference

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import org.scalatest.{FlatSpec, Matchers}

class InferenceModelSpec extends FlatSpec with Matchers {

  val resource = getClass().getClassLoader().getResource("models")
  val modelPath = resource.getPath + "/caffe/test_persist.prototxt"
  val weightPath = resource.getPath + "/caffe/test_persist.caffemodel"
  val floatInferenceModel = InferenceModelFactory.
    loadFloatInferenceModelForCaffe(modelPath, weightPath)

  "model " should "serialize" in {
    val bos = new ByteArrayOutputStream
    val out = new ObjectOutputStream(bos)
    out.writeObject(floatInferenceModel)
    out.flush()
    val bytes = bos.toByteArray()
    bos.close()

    val bis = new ByteArrayInputStream(bytes)
    val in = new ObjectInputStream(bis)
    val floatInferenceModel2 = in.readObject.asInstanceOf[FloatInferenceModel]
    // println(floatInferenceModel2.predictor)
    assert(floatInferenceModel.model == floatInferenceModel2.model)
    assert(floatInferenceModel.predictor != null)
    in.close()
  }
}
