package com.intel.analytics.bigdl.apps.recommendation.inference

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.orca.inference.InferenceModel

class NueralCFModel extends InferenceModel{

  def load(path:String) = {
    doLoad(path,null)
  }

  def predict(feature:Activity)={
    doPredict(feature)
  }

  def preProcess(userItemPair:Seq[(Int, Int)]) = {

    userItemPair.map(x=> {

      val feature: Tensor[Float] =Tensor(T(T(x._1, x._2)))

      (x, feature)

    })

  }
}
