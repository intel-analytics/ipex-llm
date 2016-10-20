/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.models.cifar

import com.intel.analytics.sparkdl.nn.{LogSoftMax, _}
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object VggLike {
  def apply[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[Tensor[T], Tensor[T], T] = {
    val vggBnDo = new Sequential[Tensor[T], Tensor[T], T]()
    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int): Sequential[Tensor[T], Tensor[T], T] = {
      vggBnDo.add(new SpatialConvolution[T](nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(new SpatialBatchNormalization[T](nOutPutPlane, 1e-3))
      vggBnDo.add(new ReLU[T](true))
      vggBnDo
    }
    convBNReLU(3, 64).add(new Dropout[T]((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(64, 128).add(new Dropout[T](0.4))
    convBNReLU(128, 128)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256).add(new Dropout[T](0.4))
    convBNReLU(256, 256)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())

    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512).add(new Dropout[T](0.4))
    convBNReLU(512, 512)
    vggBnDo.add(new SpatialMaxPooling[T](2, 2, 2, 2).ceil())
    vggBnDo.add(new View[T](512))

    val classifier = new Sequential[Tensor[T], Tensor[T], T]()
    classifier.add(new Dropout[T](0.5))
    classifier.add(new Linear[T](512, 512))
    classifier.add(new BatchNormalization[T](512))
    classifier.add(new ReLU[T](true))
    classifier.add(new Dropout[T](0.5))
    classifier.add(new Linear[T](512, classNum))
    classifier.add(new LogSoftMax[T])
    vggBnDo.add(classifier)

    vggBnDo
  }
}
