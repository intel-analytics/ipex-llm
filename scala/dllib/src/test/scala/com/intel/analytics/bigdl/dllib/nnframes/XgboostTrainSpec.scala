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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.sql.{SQLContext, SparkSession}

class XgboostTrainSpec extends ZooSpecHelper {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _

  override def doBefore(): Unit = {
    val conf = Engine.createSparkConf().setAppName("Test NNClassifier").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "XGBRegressor train" should "work" in {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val df = Seq(
      (1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f, 4.0f, 8.0f, 3.0f, 116.3668f),
      (1.0f, 3.0f, 8.0f, 6.0f, 5.0f, 9.0f, 5.0f, 6.0f, 7.0f, 4.0f, 116.367f),
      (2.0f, 1.0f, 5.0f, 7.0f, 6.0f, 7.0f, 4.0f, 1.0f, 2.0f, 3.0f, 116.367f),
      (2.0f, 1.0f, 4.0f, 3.0f, 6.0f, 1.0f, 3.0f, 2.0f, 1.0f, 3.0f, 116.3668f)
    ).toDF("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "label")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"))
      .setOutputCol("features")
    val assembledDf = vectorAssembler.transform(df).select("features", "label").cache()

    val xgbRf0 = new XGBRegressor()
    val xgbRegressorModel0 = xgbRf0.fit(assembledDf)
    val y0 = xgbRegressorModel0.transform(assembledDf)

    xgbRegressorModel0.save("/tmp/test")
    val model = XGBRegressorModel.load("/tmp/test")
    val y0_0 = model.transform(assembledDf)
    assert(y0_0.except(y0).count()==0)
  }
}
