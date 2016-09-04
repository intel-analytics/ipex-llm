package com.intel.analytics.dllib.lib.optim

import com.intel.analytics.dllib.lib.nn.Module
import com.intel.analytics.dllib.lib.tensor.torch

trait ModelPersist[@specialized(Float, Double) T] {

  private var modelSaveInterval : Option[Int] = None

  private var path : Option[String] = None

  private var isOverWrite = true

  def setModelSaveInterval(modelSaveInterval : Int) : this.type = {
    require(modelSaveInterval > 0)
    this.modelSaveInterval = Some(modelSaveInterval)
    this
  }

  def setPath(path : String) : this.type = {
    if(path != null) {
      this.path = Some(path)
    }
    this
  }

  def setOverWrite(isOverWrite : Boolean) : this.type = {
    this.isOverWrite = isOverWrite
    this
  }


  def saveModel(model : Module[T], iter : Int, force : Boolean = false) : this.type = {
    if(this.path.isDefined) {
      require(model != null)

      if (force) {
        torch.saveObj(model, path.get, isOverWrite)
      } else if (modelSaveInterval.isDefined && iter % modelSaveInterval.get == 0) {
        torch.saveObj(model, s"$path.$iter", isOverWrite)
      }
    }

    this
  }

  def saveModel(model : Module[T]) : this.type = {
    saveModel(model, 0, true)
  }

}
