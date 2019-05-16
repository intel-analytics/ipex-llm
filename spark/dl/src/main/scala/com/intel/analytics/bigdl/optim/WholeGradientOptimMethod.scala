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

package com.intel.analytics.bigdl.optim

/**
 * Extending this trait means that the OptimMethod requires fetching the whole gradient
 */
trait WholeGradientOptimMethod {

}

object WholeGradientOptimMethod {
  /**
   * Check if the OptimMethod requires fetching the whole gradient
   * @param optim the optim method to check
   * @return true if the optim method requires fetching the whole gradient
   */
  def checkRequiresWholeGradient[T](optim: OptimMethod[T]): Boolean = {
    optim.isInstanceOf[WholeGradientOptimMethod]
  }

  /**
   * Check if the OptimMethods require fetching the whole gradient
   * @param optims the map from layer names to optim methods
   * @return true if any of the optim methods requires fetching the whole gradient
   */
  def checkRequiresWholeGradient[T](optims: Map[String, OptimMethod[T]]): Boolean = {
    var result = false
    optims.foreach( kv => result |= checkRequiresWholeGradient(kv._2))
    result
  }
}
