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

object KerasRunner {
  // scalastyle:off
  val code_head =
    """
      |from keras.layers.core import *
      |from keras.layers.convolutional import *
      |from keras.layers import *
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
  val code_bottom =
    """
      |nb_sample= 1
      |input_shape = input_tensor.shape.as_list()
      |input_shape[0] = nb_sample
      |X = np.random.uniform(0, 1, input_shape)
      |
      |grad_input = K.get_session().run(K.gradients(model.output, model.input), feed_dict={input_tensor: X}) # grad_input
      |
      |grad_weight = K.get_session().run(K.gradients(model.output, model.trainable_weights),  # grad_weight
      |                        feed_dict={input_tensor: X})
      |weights = model.get_weights()
      |output = model.predict(X)
      |result_list = []
      |for item in [("weights", weights), ("input", X), ("grad_input", grad_input), ("grad_weight", grad_weight), ("output",output)]:
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
  // scalastyle:on
  private def getWeightRelate(pvalues: Map[String, Array[Float]],
                              keyName: String): Array[Tensor[Float]] = {
    if (!pvalues.keySet.filter(key => key.contains(keyName)).isEmpty) {
      val weightNum = pvalues.keySet.filter(key => key.contains(keyName)).size / 2
      Range(0, weightNum).map {i =>
        Tensor[Float](
          data = pvalues(s"${keyName}_${i}_value"),
          shape = pvalues(s"${keyName}_${i}_shape").map(_.toInt))
      }.toArray
    } else {
      null
    }
  }

  private def getNoneWeightRelate(pvalues: Map[String, Array[Float]],
                                  keyName: String): Tensor[Float] = {
    Tensor[Float](
      data = pvalues(s"${keyName}_value"),
      shape = pvalues(s"${keyName}_shape").map(_.toInt))
  }

  // return: (grad_input, grad_weight, weights, input, output)
  def run(code: String): (Tensor[Float], Array[Tensor[Float]],
    Array[Tensor[Float]], Tensor[Float], Tensor[Float]) = {
    val pcodeFile = java.io.File.createTempFile("UnitTest", "keras")
    val writer = new PrintWriter(pcodeFile)
    writer.write(code_head)
    writer.write(code)
    writer.write(code_bottom)
    writer.close()
    val pcodeFileAbsPath = pcodeFile.getAbsolutePath
    println("python code file: " + pcodeFileAbsPath)
    val resultPaths = s"python ${pcodeFileAbsPath}".!!.split("\n")
    if (pcodeFile.exists()) {
      pcodeFile.delete()
    }

    val pvalues = resultPaths.map {file =>
      val value = Source.fromFile(file).getLines().map(_.toFloat).toArray
      val key = file.split("-")(2)
      key -> value
    }.toMap

    resultPaths.foreach {path =>
      new File(path).delete()
    }

    val grad_input = getNoneWeightRelate(pvalues, "grad_input")

    val grad_weight = getWeightRelate(pvalues, "grad_weight")

    val weights = getWeightRelate(pvalues, "weights")

    val input = getNoneWeightRelate(pvalues, "input")

    val output = getNoneWeightRelate(pvalues, "output")

    (grad_input, grad_weight, weights, input, output)
  }

}
