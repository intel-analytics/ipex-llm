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
package org.apache.spark.ml.param

import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.OptimMethodParam._
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.reflect.ClassTag

/**
 * :: DeveloperApi ::
 * A param wrapper for [[com.intel.analytics.bigdl.optim.OptimMethod]]
 */
@DeveloperApi
class OptimMethodParam(
  parent: Params, name: String, doc: String, isValid: OptimMethod[_] => Boolean
) extends Param[OptimMethod[_]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, ParamValidators.alwaysTrue)

  override def jsonDecode(json: String): OptimMethod[_] = {
    implicit val formats = DefaultFormats
    val js = parse(json)
    (js \ "tensorDataType").extract[String] match {
      case "TensorDouble" =>
        decode[Double]((js \ "objectData").asInstanceOf[JObject])
      case "TensorFloat" =>
        decode[Float]((js \ "objectData").asInstanceOf[JObject])
      case tt =>
        throw new IllegalArgumentException(s"Wrong TensorDataType: $tt")
    }
  }

  override def jsonEncode(value: OptimMethod[_]): String = {
    // infer tensor data type by suffix of ClassName
    val json = value.getClass.getName match {
      case clazz if clazz.endsWith("mcD$sp") =>
        val jObject = encode[Double](value.asInstanceOf[OptimMethod[Double]])
        ("objectData" -> jObject) ~ ("tensorDataType" -> "TensorDouble")
      case clazz if clazz.endsWith("mcF$sp") =>
        val jObject = encode[Float](value.asInstanceOf[OptimMethod[Float]])
        ("objectData" -> jObject) ~ ("tensorDataType" -> "TensorFloat")
      case clazz =>
        throw new IllegalArgumentException(s"Wrong ClassName: $clazz")
    }
    compact(render(json))
  }
}

object OptimMethodParam {
  private[ml] def decode[T: ClassTag](json: JObject
  )(implicit ev: TensorNumeric[T]): OptimMethod[T] = {
    implicit val formats = DefaultFormats
    val id = (json \ "optId").extract[String]
    val opt = findDecoder(id).decode[T](json)
    opt
  }

  private[ml] def encode[T: ClassTag](opt: OptimMethod[T]
  )(implicit ev: TensorNumeric[T]): JObject = {
    val builder = findEncoder(opt)
    builder.encode[T](opt) ~ ("optId" -> builder.uid)
  }

  private def findDecoder(id: String): OptimMethodBuilder = {
    id match {
      case AdadeltaBuilder.uid => AdadeltaBuilder
      case AdagradBuilder.uid => AdagradBuilder
      case AdamBuilder.uid => AdamBuilder
      case AdamaxBuilder.uid => AdamaxBuilder
      case RMSpropBuilder.uid => RMSpropBuilder
      case LBFGSBuilder.uid => LBFGSBuilder
      case SGDBuilder.uid => SGDBuilder
    }
  }

  private def findEncoder[T: ClassTag](opt: OptimMethod[T]): OptimMethodBuilder = {
    opt match {
      case _: Adadelta[T] => AdadeltaBuilder
      case _: Adagrad[T] => AdagradBuilder
      case _: Adam[T] => AdamBuilder
      case _: Adamax[T] => AdamaxBuilder
      case _: RMSprop[T] => RMSpropBuilder
      case _: LBFGS[T] => LBFGSBuilder
      case _: SGD[T] => SGDBuilder
    }
  }
}

trait OptimMethodBuilder extends Serializable {
  val uid: String
  def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): OptimMethod[T]
  def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject
}

object AdadeltaBuilder extends OptimMethodBuilder {
  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): Adadelta[T] = {
    implicit val formats = DefaultFormats
    val decayRate = (json \ "decayRate").extract[Double]
    val epsilon = (json \ "Epsilon").extract[Double]
    val adadelta = if (ev.getType() == DoubleType) {
      new Adadelta[Double](decayRate, epsilon)
    } else if (ev.getType() == FloatType) {
      new Adadelta[Float](decayRate, epsilon)
    }
    adadelta.asInstanceOf[Adadelta[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[Adadelta[T]], "Wrong OptimMethod input: it should be Adadelta.")
    val adaDelta = opt.asInstanceOf[Adadelta[T]]
    ("decayRate" -> adaDelta.decayRate) ~ ("Epsilon" -> adaDelta.Epsilon)
  }

  override val uid: String = "AdadeltaOpt"
}

object AdagradBuilder extends OptimMethodBuilder {
  override val uid: String = "AdagradOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): Adagrad[T] = {
    implicit val formats = DefaultFormats
    val learningRate = (json \ "learningRate").extract[Double]
    val learningRateDecay = (json \ "learningRateDecay").extract[Double]
    val weightDecay = (json \ "weightDecay").extract[Double]
    val adadgrad = if (ev.getType() == DoubleType) {
      new Adagrad[Double](learningRate, learningRateDecay, weightDecay)
    } else if (ev.getType() == FloatType) {
      new Adagrad[Float](learningRate, learningRateDecay, weightDecay)
    }
    adadgrad.asInstanceOf[Adagrad[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[Adagrad[T]], "Wrong OptimMethod input: it should be Adagrad.")
    val inst = opt.asInstanceOf[Adagrad[T]]
    ("learningRate" -> inst.learningRate) ~
      ("learningRateDecay" -> inst.learningRateDecay) ~
      ("weightDecay" -> inst.weightDecay)
  }
}

object AdamBuilder extends OptimMethodBuilder {
  override val uid: String = "AdamOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): Adam[T] = {
    implicit val formats = DefaultFormats
    val learningRate = (json \ "learningRate").extract[Double]
    val learningRateDecay = (json \ "learningRateDecay").extract[Double]
    val beta1 = (json \ "beta1").extract[Double]
    val beta2 = (json \ "beta2").extract[Double]
    val epsilon = (json \ "Epsilon").extract[Double]
    val adam = if (ev.getType() == DoubleType) {
      new Adam[Double](learningRate, learningRateDecay, beta1, beta2, epsilon)
    } else if (ev.getType() == FloatType) {
      new Adam[Float](learningRate, learningRateDecay, beta1, beta2, epsilon)
    }
    adam.asInstanceOf[Adam[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[Adam[T]], "Wrong OptimMethod input: it should be Adam.")
    val inst = opt.asInstanceOf[Adam[T]]
    ("learningRate" -> inst.learningRate) ~
      ("learningRateDecay" -> inst.learningRateDecay) ~
      ("beta1" -> inst.beta1) ~
      ("beta2" -> inst.beta2) ~
      ("Epsilon" -> inst.Epsilon)
  }
}

object AdamaxBuilder extends OptimMethodBuilder {
  override val uid: String = "AdamaxOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): Adamax[T] = {
    implicit val formats = DefaultFormats
    val learningRate = (json \ "learningRate").extract[Double]
    val beta1 = (json \ "beta1").extract[Double]
    val beta2 = (json \ "beta2").extract[Double]
    val epsilon = (json \ "Epsilon").extract[Double]
    val adamax = if (ev.getType() == DoubleType) {
      new Adamax[Double](learningRate, beta1, beta2, epsilon)
    } else if (ev.getType() == FloatType) {
      new Adamax[Float](learningRate, beta1, beta2, epsilon)
    }
    adamax.asInstanceOf[Adamax[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[Adamax[T]], "Wrong OptimMethod input: it should be Adamax.")
    val inst = opt.asInstanceOf[Adamax[T]]
    ("learningRate" -> inst.learningRate) ~
      ("beta1" -> inst.beta1) ~
      ("beta2" -> inst.beta2) ~
      ("Epsilon" -> inst.Epsilon)
  }
}

object RMSpropBuilder extends OptimMethodBuilder {
  override val uid: String = "RMSpropOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): RMSprop[T] = {
    implicit val formats = DefaultFormats
    val learningRate = (json \ "learningRate").extract[Double]
    val learningRateDecay = (json \ "learningRateDecay").extract[Double]
    val decayRate = (json \ "decayRate").extract[Double]
    val epsilon = (json \ "Epsilon").extract[Double]
    val rMSprop = if (ev.getType() == DoubleType) {
      new RMSprop[Double](learningRate, learningRateDecay, decayRate, epsilon)
    } else if (ev.getType() == FloatType) {
      new RMSprop[Float](learningRate, learningRateDecay, decayRate, epsilon)
    }
    rMSprop.asInstanceOf[RMSprop[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[RMSprop[T]], "Wrong OptimMethod input: it should be RMSprop.")
    val inst = opt.asInstanceOf[RMSprop[T]]
    ("learningRate" -> inst.learningRate) ~
      ("learningRateDecay" -> inst.learningRateDecay) ~
      ("decayRate" -> inst.decayRate) ~
      ("Epsilon" -> inst.Epsilon)
  }
}

/**
 * So far, serialization/deserialization for
 * Params[lineSearch, lineSearchOptions] are not supported.
 */
object LBFGSBuilder extends OptimMethodBuilder {
  override val uid: String = "LBFGSOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): LBFGS[T] = {
    implicit val formats = DefaultFormats
    val maxIter = (json \ "maxIter").extract[Int]
    val maxEval = (json \ "maxEval").extract[Double]
    val tolFun = (json \ "tolFun").extract[Double]
    val tolX = (json \ "tolX").extract[Double]
    val nCorrection = (json \ "nCorrection").extract[Int]
    val learningRate = (json \ "learningRate").extract[Double]
    val verbose = (json \ "verbose").extract[Boolean]
    val lbfgs = if (ev.getType() == DoubleType) {
      new LBFGS[Double](maxIter, maxEval, tolFun, tolX, nCorrection, learningRate, verbose)
    } else if (ev.getType() == FloatType) {
      new LBFGS[Float](maxIter, maxEval, tolFun, tolX, nCorrection, learningRate, verbose)
    }
    lbfgs.asInstanceOf[LBFGS[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[LBFGS[T]], "Wrong OptimMethod input: it should be LBFGS.")
    val inst = opt.asInstanceOf[LBFGS[T]]
    ("maxIter" -> inst.maxIter) ~
      ("maxEval" -> inst.maxEval) ~
      ("tolFun" -> inst.tolFun) ~
      ("tolX" -> inst.tolX) ~
      ("nCorrection" -> inst.nCorrection) ~
      ("learningRate" -> inst.learningRate) ~
      ("verbose" -> inst.verbose)
  }
}

/**
 * So far, serialization/deserialization for
 * Params[learningRateSchedule, learningRates, weightDecays] are not supported.
 */
object SGDBuilder extends OptimMethodBuilder {
  override val uid: String = "SGDOpt"

  override def decode[T: ClassTag](json: JObject)(implicit ev: TensorNumeric[T]): SGD[T] = {
    implicit val formats = DefaultFormats
    val learningRate = (json \ "learningRate").extract[Double]
    val learningRateDecay = (json \ "learningRateDecay").extract[Double]
    val weightDecay = (json \ "weightDecay").extract[Double]
    val momentum = (json \ "momentum").extract[Double]
    val dampening = (json \ "dampening").extract[Double]
    val nesterov = (json \ "nesterov").extract[Boolean]
    val sgd = if (ev.getType() == DoubleType) {
      new SGD[Double](learningRate, learningRateDecay, weightDecay, momentum, dampening, nesterov)
    } else if (ev.getType() == FloatType) {
      new SGD[Float](learningRate, learningRateDecay, weightDecay, momentum, dampening, nesterov)
    }
    sgd.asInstanceOf[SGD[T]]
  }

  override def encode[T: ClassTag](opt: OptimMethod[T])(implicit ev: TensorNumeric[T]): JObject = {
    require(opt.isInstanceOf[SGD[T]], "Wrong OptimMethod input: it should be SGD.")
    val inst = opt.asInstanceOf[SGD[T]]
    ("learningRate" -> inst.learningRate) ~
      ("learningRateDecay" -> inst.learningRateDecay) ~
      ("weightDecay" -> inst.weightDecay) ~
      ("momentum" -> inst.momentum) ~
      ("dampening" -> inst.dampening) ~
      ("nesterov" -> inst.nesterov)
  }
}

/**
 * :: DeveloperApi ::
 * A param wrapper for [[com.intel.analytics.bigdl.optim.Trigger]]
 */
@DeveloperApi
class TriggerParam(
  parent: Params, name: String, doc: String, isValid: Trigger => Boolean
) extends Param[Trigger](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, ParamValidators.alwaysTrue)

  /**
   * Since [[Trigger]] doesn't contain any Type Param,
   * it's safe to be serialized/deserialzed by Java Serialization.
   * What's more, [[Trigger]] are implemented by anonymous classes whose params are hard to fetch.
   */
  override def jsonDecode(str: String): Trigger = {
    val bytes = str.split(",").map(_.toByte)
    SerializationUtils.deserialize[Trigger](bytes)
  }

  override def jsonEncode(value: Trigger): String = {
    val bytes = SerializationUtils.serialize(value)
    bytes.mkString(",")
  }
}
