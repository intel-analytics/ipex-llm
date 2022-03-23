/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.pytorch

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.python.api.JTensor
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.MetaData
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils._
import com.intel.analytics.bigdl.ppml.utils.{FLClientClosable, ProtoUtils}

object PytorchSuite extends FLClientClosable {
  /**
   * Convert Python tensor to scala, upload to FLServer, get response and convert back to Python
   * @param pred
   * @param target
   * @param version
   * @param algorithm
   * @return
   */
  def trainStep(pred: JTensor, target: JTensor, version: Int, algorithm: String) = {
    // convert JTensor to Proto
    val metadata = MetaData.newBuilder
      .setName(s"pytorch_output").setVersion(version).build
    val tableProto = outputTargetToTableProto(
      Tensor[Float](pred.storage, pred.shape),
      Tensor[Float](target.storage, target.shape),
      metadata)
    val result = flClient.nnStub.train(tableProto, algorithm).getData
    val errors = getTensor("gradInput", result)
    JTensor(errors.storage().array(), errors.size(), "float")
  }
}
