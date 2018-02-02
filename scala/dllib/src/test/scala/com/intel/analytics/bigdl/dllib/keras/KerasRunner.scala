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
package com.intel.analytics.bigdl.keras

import java.io.{File, PrintWriter}

import com.intel.analytics.bigdl.tensor.Tensor

import scala.io.Source
import scala.sys.process._

sealed trait MainCodeType
object Loss extends MainCodeType
object Layer extends MainCodeType
object Regularizer extends MainCodeType

object KerasRunner {
  // scalastyle:off
  val code_head =
    """
      |from keras.layers.core import *
      |from keras.layers.convolutional import *
      |from keras.layers import *
      |from keras.objectives import *
      |from keras.regularizers import *
      |from keras.models import Model
      |import keras.backend as K
      |import numpy as np
      |import tempfile
      |
      |np.random.seed(1337) # for reproducibility
      |
      |def create_tmp_path(name):
      |    tmp_file = tempfile.NamedTemporaryFile(prefix="UnitTest-keras-" + name + "-")
      |    tmp_file.close()
      |    return tmp_file.name
      |
    """.stripMargin

  val code_for_loss =
  """
    |grad_input = K.get_session().run(K.gradients(loss, [input_tensor]),
    |                           feed_dict={input_tensor: input, target_tensor: Y})
    |output = K.get_session().run(loss, feed_dict={input_tensor: input, target_tensor: Y})
    |weights = []
    |grad_weight = []
  """.stripMargin
  val code_for_layer =
    """
      |Y = []
      |output = model.predict(input)
      |
      |grad_input = K.get_session().run(K.gradients(model.output * output, model.input), feed_dict={input_tensor: input}) # grad_input
      |
      |grad_weight = K.get_session().run(K.gradients(model.output * output, model.trainable_weights),  # grad_weight
      |                        feed_dict={input_tensor: input})
      |weights = model.get_weights()

      """.stripMargin
  val code_for_save = """
      |result_list = []
      |for item in [("weights", weights), ("input", input), ("target", Y), ("grad_input", grad_input), ("grad_weight", grad_weight), ("output",output)]:
      |    if isinstance(item[1], list):
      |        if len(item[1]) > 1:
      |            for i in range(len(item[1])):
      |                result_list.append((item[0] + "_" + str(i), item[1][i]))
      |        elif len(item[1]) == 1:
      |            result_list.append((item[0], item[1][0]))
      |        else:
      |            continue
      |    else:
      |        result_list.append(item)
      |for result in result_list:
      |    value_path = create_tmp_path(result[0] + "_value")
      |    shape_path = create_tmp_path(result[0] + "_shape")
      |    np.savetxt(shape_path, result[1].shape)
      |    np.savetxt(value_path, result[1].ravel())
      |    print(shape_path)
      |    print(value_path)
      |
      |
    """.stripMargin

  val code_for_regularizer =
    """
      |Y = K.get_session().run(model.losses, feed_dict={input_tensor: input})
      |output = model.predict(input)
      |grad_input = K.get_session().run(K.gradients(model.losses, [input_tensor]),
      |                           feed_dict={input_tensor: input})
      |grad_input += output # they're two branches, we should gather them.
      |weights = []
      |grad_weight = []
    """.stripMargin

  // scalastyle:on
  private def getWeightRelate(pvalues: Map[String, Array[Float]],
                              keyName: String): Array[Tensor[Float]] = {
    if (!pvalues.keySet.filter(key => key.contains(keyName)).isEmpty) {
      val weightNum = pvalues.keySet.filter(key => key.contains(keyName)).size / 2
      Range(0, weightNum).map {i =>
        val keyPrefix = if (weightNum > 1) keyName + "_" + i else keyName
        Tensor[Float](
          data = pvalues(s"${keyPrefix}_value"),
          shape = pvalues(s"${keyPrefix}_shape").map(_.toInt))
      }.toArray
    } else {
      null
    }
  }

  private def getNoneWeightRelate(pvalues: Map[String, Array[Float]],
                                  keyName: String): Tensor[Float] = {
    if (!pvalues.keySet.filter(key => key.contains(keyName)).isEmpty) {
      Tensor[Float](
        data = pvalues(s"${keyName}_value"),
        shape = pvalues(s"${keyName}_shape").map(_.toInt))
    } else {
      null
    }
  }

  // return: (grad_input, grad_weight, weights, input, target, output)
  def run(code: String, codeType: MainCodeType = Layer): (Tensor[Float], Array[Tensor[Float]],
    Array[Tensor[Float]], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    val pcodeFile = java.io.File.createTempFile("UnitTest", "keras")
    val writer = new PrintWriter(pcodeFile)
    writer.write(code_head)
    writer.write(code)
    writer.write(
      codeType match {
        case Layer => code_for_layer
        case Loss => code_for_loss
        case Regularizer => code_for_regularizer
      })
    writer.write(code_for_save)
    writer.close()
    val pcodeFileAbsPath = pcodeFile.getAbsolutePath
    println("python code file: " + pcodeFileAbsPath)
    val resultPaths = s"python ${pcodeFileAbsPath}".!!.split("\n")

    val pvalues = resultPaths.map {file =>
      val value = Source.fromFile(file).getLines().map(_.toFloat).toArray
      val key = file.split("-")(2)
      key -> value
    }.toMap

    val grad_input = getNoneWeightRelate(pvalues, "grad_input")

    val grad_weight = getWeightRelate(pvalues, "grad_weight")

    val weights = getWeightRelate(pvalues, "weights")

    val input = getNoneWeightRelate(pvalues, "input")

    val target = getNoneWeightRelate(pvalues, "target")

    var output = getNoneWeightRelate(pvalues, "output")

    resultPaths.foreach {path =>
      new File(path).delete()
    }
    if (pcodeFile.exists()) {
      pcodeFile.delete()
    }

    (grad_input, grad_weight, weights, input, target, output)
  }

}
