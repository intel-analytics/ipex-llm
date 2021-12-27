package com.intel.analytics.bigdl.serving.models

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper}
import org.scalatest.{FlatSpec, Matchers}

class BigDLModelSpec extends FlatSpec with Matchers {
  "BigDL NNModel" should "work" in {
    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "bigdl"
    helper.weightPath = "/home/litchy/models/linear.model"
    ClusterServing.model = helper.loadInferenceModel()
    val tensor = Tensor[Float](1, 2).rand()
    val result = ClusterServing.model.doPredict(tensor)
    require(result.toTensor[Float].size().sameElements(Array(1, 1)), "shape error")
  }
}
