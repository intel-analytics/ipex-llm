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
package com.intel.analytics.zoo.feature

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.common.{BigDLAdapter, Preprocessing}
import com.intel.analytics.zoo.feature.image.{ImageResize, ImageSet}
import org.apache.spark.SparkConf
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class FeatureSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val resource = getClass.getClassLoader.getResource("imagenet/n04370456/")

  "BigDLAdapter" should "adapt BigDL Transformer" in {
    val newResize = BigDLAdapter(ImageResize(1, 1))
    assert(newResize.isInstanceOf[Preprocessing[_, _]])
  }

  "Local ImageSet" should "work with resize" in {
    val image = ImageSet.read(resource.getFile, resizeH = 200, resizeW = 200)
    val imf = image.toLocal().array.head
    require(imf.getHeight() == 200)
    require(imf.getWidth() == 200)
  }

  "Distribute ImageSet" should "work with resize" in {
    val conf = new SparkConf().setAppName("Feature Test").setMaster("local[1]")
    val sc = NNContext.initNNContext(conf)
    val image = ImageSet.read(resource.getFile, sc, resizeH = 200, resizeW = 200)
    val imf = image.toDistributed().rdd.collect().head
    require(imf.getHeight() == 200)
    require(imf.getWidth() == 200)
    sc.stop()
  }
}
