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

package com.intel.analytics.bigdl.example.mkldnn.int8

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.models.resnet.ImageNetDataSet
import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * GenerateInt8Scales will generate a model with scales information,
 * which will be used with mkldnn int8. You can pass a model trained from BigDL
 * and will genereate a model whose name is the same except including "quantized"
 */
object GenerateInt8Scales {
  val logger: Logger = Logger.getLogger(getClass)
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  import Utils._

  def genereateInt8Scales(model: Graph[Float], modelName: String,
    evaluationSet: RDD[MiniBatch[Float]]): Unit = {
    model.evaluate()

    model.setInputDimMask(0, true)
    model.setOutputDimMask(0, true)
    model.setWeightDimMask(1, true)

    logger.info(s"Generate the scales for $modelName ...")
    val samples = evaluationSet
      .repartition(1) // repartition (shuffle) will have better accuracy
      .take(1) // only split one batch to sample
      .map(_.getInput().toTensor[Float])

    samples.foreach { sample =>
      model.forward(sample)
      model.calcScales(sample)
    }

    // we should clean the state, such as output
    model.clearState()

    logger.info(s"Generate the scales for $modelName done.")
  }

  def saveQuantizedModel(model: Graph[Float], modelName: String): Unit = {
    val suffix = ".bigdl"
    val prefix = modelName.stripSuffix(suffix)
    val name = prefix.concat(".quantized").concat(suffix)
    logger.info(s"Save the quantized model $name ...")
    // it will force overWrite the existed model file
    model.saveModule(name, overWrite = true)
    logger.info(s"Save the quantized model $name done.")
  }

  def main(args: Array[String]): Unit = {
    genInt8ScalesParser.parse(args, GenInt8ScalesParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Quantize the model")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val partitionNum = Engine.nodeNumber()
      val imageFrame = DataSet.SeqFileFolder.filesToImageFrame(param.folder, sc, 1000,
        partitionNum = Option(partitionNum))

      // the transformer is the same as as that in validation during training
      val evaluationSet = ImageNetDataSet.valDataSet(param.folder,
        sc, 224, param.batchSize).toDistributed().data(train = false)
      // Currently, we only support the graph model, so we add a `toGraph`
      // if the model is already graph, you can need not to it.
      val model = Module.loadModule[Float](param.model).toGraph()
      genereateInt8Scales(model, param.model, evaluationSet)
      saveQuantizedModel(model, param.model)
    }
  }
}
