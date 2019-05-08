package com.intel.analytics.zoo.apps.recommendation.inference


object SimpleScalaDriver {

  def main(args: Array[String]): Unit = {

    val modelPath = System.getProperty("MODEL_PATH", "./models/ncf.bigdl")

    val rcm = new NueralCFModel()
    rcm.load(modelPath)

    val userItemPair = ( 1 to 10).map( x=> (x, x+1))

    val userItemFeature = rcm.preProcess(userItemPair)

    userItemFeature.map(x=> {
      val r = rcm.predict(x._2)
      println(x._1 +":" + r)
    })
  }
}
