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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import play.api.libs.json._
import play.api.libs.functional.syntax._

import scala.reflect.ClassTag


//case class KerasHistory(layerName: String, nodeIndex: String, tensorIndex: String)


case class Layer(className: String,
                 config: JsValue,
                 inboundNodes: Seq[Seq[JsArray]],
                 name: String) // name is layer name
case class ModelConfig(name: String,
                       layers: Seq[Layer],
                       inputLayers: Seq[JsArray],
                       outputLayers: Seq[JsArray]) // name is model name
case class KModel(className: String, config: ModelConfig, kerasVersion: String)

class BaseLayerConfig(val name: String,
                      val trainable: Boolean,
                      val batchInputShape: Option[Array[String]],
                      val inputDtype: Option[String]) {
  def this(config: JsValue) = {
    this(
      (JsPath \ "name").read[String].reads(config).get,
      (JsPath \ "trainable").read[Boolean].reads(config).get,
      (JsPath \ "batch_input_shape")
        .readNullable[JsArray]
        .reads(config).get
        .map {jarray =>
          jarray.value.toList.map(_.as[String]).toArray
        },
      (JsPath \ "input_dtype").readNullable[String].reads(config).get
    )
  }
}

class InputConfig(config: JsValue) extends BaseLayerConfig(config) {
  val sparse: Boolean = (JsPath \ "sparse").read[Boolean].reads(config).get
}

class FlattenConfig(config: JsValue) extends BaseLayerConfig(config)

class EmbeddingConfig(config: JsValue) extends BaseLayerConfig(config) {
  
}

class ParametersLayerConfig(config: JsValue) extends BaseLayerConfig(config) {
  val activation = (JsPath \ "activation").read[String].reads(config).get

  val activityRegularizer = (JsPath \ "activity_regularizer").read[JsValue].reads(config).get

  val wRegularizer = (JsPath \ "W_regularizer").read[JsValue].reads(config).get
  val bRegularizer = (JsPath \ "b_regularizer").readNullable[JsValue].reads(config).get

  val bConstraint = (JsPath \ "b_constraint").readNullable[JsValue].reads(config).get
  val wConstraint = (JsPath \ "W_constraint").read[JsValue].reads(config).get
  val bias = (JsPath \ "bias").readNullable[Boolean].reads(config).get

  val initMethod = (JsPath \ "init").read[String].reads(config).get

  checkConstraint(this)
  private def checkConstraint(config: ParametersLayerConfig) = {
    if (config.wConstraint != JsNull || config.bConstraint.getOrElse(JsNull) != JsNull ) {
      throw new IllegalArgumentException("Haven't support constraint yet")
    }
  }
}

class Convolution2DConfig(config: JsValue) extends ParametersLayerConfig(config) {
  val dimOrder = (JsPath \ "dim_ordering").read[String].reads(config).get
  val nbCol = (JsPath \ "nb_col").read[Int].reads(config).get
  val nbRow = (JsPath \ "nb_row").read[Int].reads(config).get
  val subsample = (JsPath \ "subsample").read[JsArray].reads(config).get
  val nbFilter = (JsPath \ "nb_filter").read[Int].reads(config).get
  val borderMode = (JsPath \ "border_mode").read[String].reads(config).get
}

class DenseConfig(config: JsValue) extends ParametersLayerConfig(config) {
  val outputDim = (JsPath \ "output_dim").read[Int].reads(config).get
  val inputDim = (JsPath \ "input_dim").read[Int].reads(config).get
}

class ActivationConfig(config: JsValue) extends BaseLayerConfig(config) {
  val activation = (JsPath \ "activation").read[String].reads(config).get
}

class DropoutConfig(config: JsValue) extends BaseLayerConfig(config) {
  val p = (JsPath \ "p").read[Double].reads(config).get
}


class Keras1DefinitionParser[K: ClassTag] {
  private implicit val layerReads: Reads[Layer] = (
    (JsPath \ "class_name").read[String] and
      (JsPath \ "config").read[JsValue] and
      (JsPath \ "inbound_nodes").read[Seq[Seq[JsArray]]] and
      (JsPath \ "name").read[String]
    )(Layer.apply _)

  private implicit val modelConfigReads: Reads[ModelConfig] = (
    (JsPath \ "name").read[String] and
      (JsPath \ "layers").read[Seq[Layer]] and
      (JsPath \ "input_layers").read[Seq[JsArray]] and
      (JsPath \ "output_layers").read[Seq[JsArray]]
    )(ModelConfig.apply _)

  private implicit val KerasJsonReads: Reads[KModel] = (
    (JsPath \ "class_name").read[String] and
      (JsPath \ "config").read[ModelConfig] and
      (JsPath \ "keras_version").read[String]
    )(KModel.apply _)

  def parseLayer(jsonString: String): Layer = {
    val jsonObject = Json.parse(jsonString)
    val placeResult = jsonObject.validate[Layer]
    placeResult match {
      case JsSuccess(value, path) => value
    }
  }

  def parseModel(jsonString: String): KModel = {
    val jsonObject = Json.parse(jsonString)
    val placeResult = jsonObject.validate[KModel]
    placeResult match {
      case JsSuccess(value, path) => value
    }
  }
}