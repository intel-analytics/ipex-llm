package com.intel.analytics.bigdl.serving.models

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper}
import org.scalatest.{FlatSpec, Matchers}
import scala.sys.process._

class BigDLModelSpec extends FlatSpec with Matchers {
  "BigDL NNModel" should "work" in {
    ("wget --no-check-certificate -O /tmp/linear.model https://sourceforge.net/" +
      "projects/analytics-zoo/files/analytics-zoo-data/linear.model").!
    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "bigdl"
    helper.weightPath = "/tmp/linear.model"
    ClusterServing.model = helper.loadInferenceModel()
    "rm /tmp/linear.model".!
    val tensor = Tensor[Float](1, 2).rand()
    val result = ClusterServing.model.doPredict(tensor)
    require(result.toTensor[Float].size().sameElements(Array(1, 1)), "shape error")
  }
}
