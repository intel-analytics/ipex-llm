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

package com.intel.analytics.bigdl.utils.keras

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader, Customizable}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

class Keras1DefinitionParserSpec extends FlatSpec with Matchers {

  "convert flatten layer" should "ok" in {
    val flattenLayerString = """
                               |      {
                               |        "class_name": "Flatten",
                               |        "config": {
                               |          "trainable": true,
                               |          "name": "flatten_3"
                               |        },
                               |        "inbound_nodes": [
                               |          [
                               |            [
                               |              "mlp_embedding_user",
                               |              0,
                               |              0
                               |            ]
                               |          ]
                               |        ],
                               |        "name": "flatten_3"
                               |      }
                             """.stripMargin
    val flattenLayer = new Keras1DefinitionParser[Layer]().parseLayer(flattenLayerString)
    flattenLayer.className should be("Flatten")
    flattenLayer.name should be("flatten_3")
    val flattenConfig = new FlattenConfig(flattenLayer.config)
    flattenConfig.trainable should be(true)
    flattenConfig.name should be("flatten_3")
  }

  "convert dense layer" should "ok" in {
    val denseLayerStr = """
                          |      {
                          |        "class_name": "Dense",
                          |        "config": {
                          |          "W_constraint": null,
                          |          "b_constraint": null,
                          |          "name": "dense_18",
                          |          "activity_regularizer": null,
                          |          "trainable": true,
                          |          "init": "glorot_uniform",
                          |          "bias": true,
                          |          "input_dim": 3,
                          |          "b_regularizer": null,
                          |          "W_regularizer": null,
                          |          "activation": "linear",
                          |          "output_dim": 2
                          |        },
                          |        "inbound_nodes": [
                          |          [
                          |            [
                          |              "dropout_12",
                          |              0,
                          |              0
                          |            ]
                          |          ]
                          |        ],
                          |        "name": "dense_18"
                          |      }
                        """.stripMargin
    val denseLayer = new Keras1DefinitionParser[DenseConfig]().parseLayer(denseLayerStr)
    denseLayer.className should be("Dense")
    denseLayer.name should be("dense_18")
    val denseConfig = new DenseConfig(denseLayer.config)
    denseConfig.trainable should be(true)
    denseConfig.outputDim should be(2)
  }

  "parse conv2D layer" should "ok" in {
    val layerStr =
      """
        |      {
        |        "class_name": "Convolution2D",
        |        "config": {
        |          "b_regularizer": null,
        |          "W_constraint": null,
        |          "b_constraint": null,
        |          "name": "convolution2d_1",
        |          "activity_regularizer": null,
        |          "trainable": true,
        |          "dim_ordering": "tf",
        |          "nb_col": 3,
        |          "subsample": [
        |            1,
        |            1
        |          ],
        |          "init": "glorot_uniform",
        |          "bias": true,
        |          "nb_filter": 64,
        |          "input_dtype": "float32",
        |          "border_mode": "same",
        |          "batch_input_shape": [
        |            null,
        |            3,
        |            256,
        |            256
        |          ],
        |          "W_regularizer": null,
        |          "activation": "linear",
        |          "nb_row": 3
        |        },
        |        "inbound_nodes": [
        |          [
        |            [
        |              "input_1",
        |              0,
        |              0
        |            ]
        |          ]
        |        ],
        |        "name": "convolution2d_1"
        |      }
      """.stripMargin

    val denseLayer = new Keras1DefinitionParser[DenseConfig]().parseLayer(layerStr)
    denseLayer.className should be("Convolution2D")
    denseLayer.name should be("convolution2d_1")
    val config = new Convolution2DConfig(denseLayer.config)
    config.nbFilter should be(64)
    config.nbCol should be(3)
  }
  "test spark" should "ok" in {

    val conf = new SparkConf().setAppName("hello").setMaster("local[*]")
      .set("spark.driver.extraClassPath",
        "/Users/lizhichao/god/BigDL/dist/lib/bigdl-0.3.0-SNAPSHOT-jar-with-dependencies.jar")
    val sc = new SparkContext(conf)
    val result = sc.parallelize(Range(0, 10)).collect()
    println(result)
  }

}
