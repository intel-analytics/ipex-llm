package com.intel.analytics.bigdl.ppml.utils

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.spark.sql.DataFrame

object TensorUtils {
  def fromDataFrame(df: DataFrame,
                    columns: Array[String]) = {
    var rowNum = 0
    val dataArray = df.collect().map(row => {
      if (rowNum == 0) rowNum = row.length
      val rowArray = new Array[Float](row.length)
      columns.indices.foreach(i => {
        rowArray(i) = row.getAs[Float](columns(i))
      })
      rowArray
    })
    Tensor[Float](dataArray.flatten, Array(rowNum, columns.length))
  }
}
