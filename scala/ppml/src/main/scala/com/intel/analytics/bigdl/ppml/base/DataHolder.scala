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

package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.{BoostEval, DataSplit, TreeLeaves}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.TensorMap


/**
 * The holder of protobuf data class, one DataHolder could have only one instance of
 * all its acceptable data types
 * @param _tensorMap [[TensorMap]] instance of protobuf
 * @param _split [[DataSplit]] instance of protobuf
 * @param _treeLeaves [[TreeLeaves]] instance of protobuf
 * @param _boostEval [[BoostEval]] instance of protobuf
 */
class DataHolder(_tensorMap: TensorMap = null,
                 _split: DataSplit = null,
                 _treeLeaves: TreeLeaves = null,
                 _boostEval: java.util.List[BoostEval] = null) {
  var tensorMap: TensorMap = null
  var split: DataSplit = null
  var treeLeaves: TreeLeaves = null
  var boostEval: java.util.List[BoostEval] = null
  if (_tensorMap != null) tensorMap = _tensorMap
  if (_split != null) split = _split
  if (_treeLeaves != null) treeLeaves = _treeLeaves
  if (_boostEval != null) boostEval = _boostEval
  def this(value: TensorMap) = this(_tensorMap = value)
  def this(value: DataSplit) = this(_split = value)
  def this(value: TreeLeaves) = this(_treeLeaves = value)
  def this(value: java.util.List[BoostEval]) = this(_boostEval = value)
}
