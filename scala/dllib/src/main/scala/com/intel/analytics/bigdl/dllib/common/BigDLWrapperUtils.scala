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

package com.intel.analytics.bigdl.nn

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.mutable.ArrayBuffer

object BigDLWrapperUtils {
  def selectCopy[T: ClassTag](src: Tensor[T], srcIndex: Int, dst: Tensor[T]): Tensor[T] = {
    Recurrent.selectCopy(src, srcIndex, dst)
  }

  def copy[T: ClassTag](src: ArrayBuffer[Tensor[T]], dst: Tensor[T]): Unit = {
    Recurrent.copy(src, dst)
  }
}
