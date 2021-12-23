package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.{BoostEval, DataSplit, TreeLeaves}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table


class DataHolder(_table: Table = null,
                 _split: DataSplit = null,
                 _treeLeaves: TreeLeaves = null,
                 _boostEval: BoostEval = null) {
  var table: Table = null
  var split: DataSplit = null
  var leaves: TreeLeaves = null
  var boostEval: BoostEval = null
  if (_table != null) table = _table
  if (_split != null) split = _split
  def this(value: Table) = this(_table = value)
  def this(value: DataSplit) = this(_split = value)
  def this(value: TreeLeaves) = this(_treeLeaves = value)
  def this(value: BoostEval) = this(_boostEval = value)
}
