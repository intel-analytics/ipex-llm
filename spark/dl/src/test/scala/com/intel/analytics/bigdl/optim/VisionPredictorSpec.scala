/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.{FloatsToTensor, ImageFeature, ImageFrame, ImageFrameToSample}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class VisionPredictorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("pascal/")
  var sc: SparkContext = null
  var sqlContext: SQLContext = null
  before {
    val conf = Engine.createSparkConf().setAppName("ImageSpec").setMaster("local[2]")
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
    Engine.init
  }

  after {
    if (null != sc) sc.stop()
  }
  "VisionPredictor" should "work" in {
    val imageFrame = ImageFrame.read(resource.getFile, sc) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      HFlip(0.5) -> ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) -> MatToFloats() ->
      FloatsToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val predictor = new VisionPredictor(model)
    val detection = predictor.predict(imageFrame)
    val feature = detection.rdd.first()
    println(feature(ImageFeature.predict))
  }
}
