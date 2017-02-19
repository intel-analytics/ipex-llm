/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.fasterrcnn.model

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.fasterrcnn.layers.{Proposal, ReshapeInfer}
import com.intel.analytics.bigdl.models.fasterrcnn.model.Model._
import com.intel.analytics.bigdl.models.fasterrcnn.model.Phase._
import com.intel.analytics.bigdl.models.fasterrcnn.utils.FileUtil
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{File => DlFile}
import org.apache.log4j.Logger

import scala.util.Random


abstract class FasterRcnn(var phase: PhaseType, classNum: Int = 21) {

  val logger = Logger.getLogger(getClass)
  val param: FasterRcnnParam

  def train(): Unit = setPhase(TRAIN)

  def evaluate(): Unit = setPhase(TEST)

  def isTrain: Boolean = phase == TRAIN

  def isTest: Boolean = phase == TEST

  private def setPhase(phase: PhaseType): Unit = this.phase = phase

  /**
   *
   * @param p    parameter: (nIn: Int, nOut: Int, ker: Int, stride: Int, pad: Int)
   * @param name name of layer
   * @return
   */
  def conv(p: (Int, Int, Int, Int, Int),
    name: String, isBack: Boolean = true,
    initMethod: InitializationMethod = Xavier,
    init: (Double, Double) = null): SpatialConvolution[Float] = {
    val module: SpatialConvolution[Float] =
      new SpatialConvolution(p._1, p._2, p._3, p._3, p._4, p._4,
        p._5, p._5, propagateBack = isBack, initMethod = initMethod).setName(name)
    if (phase == TRAIN && init != null) initParameters(module, init)
    module
  }

  /**
   *
   * @param p    (nIn, nOut)
   * @param name name of layer
   * @return
   */
  def linear(p: (Int, Int), name: String, init: (Double, Double) = null): Linear[Float] = {
    val module = new Linear(p._1, p._2).setName(name)
    if (phase == TRAIN && init != null) initParameters(module, init)
    module
  }

  private var testModel: Option[Sequential[Float]] = None
  private var trainModel: Option[Sequential[Float]] = None

  def getTestModel: Module[Float] = {
    testModel match {
      case None => testModel = Some(createTestModel())
      case _ =>
    }
    testModel.get.evaluate()
  }

  def getTrainModel: Module[Float] = {
    throw new NotImplementedError()
  }

  def criterion4: ParallelCriterion[Float]

  protected def createFeatureAndRpnNet(): Sequential[Float]

  // pool is the parameter of RoiPooling
  val pool: Int
  private[this] var _featureAndRpnNet: Sequential[Float] = _

  def featureAndRpnNet: Sequential[Float] = {
    if (_featureAndRpnNet == null) {
      _featureAndRpnNet = createFeatureAndRpnNet()
    }
    _featureAndRpnNet
  }

  def setFeatureAndRpnNet(value: Sequential[Float]): Unit = {
    _featureAndRpnNet = value
  }

  private[this] var _fastRcnn: Sequential[Float] = _

  def fastRcnn: Sequential[Float] = {
    if (_fastRcnn == null) {
      _fastRcnn = createFastRcnn()
    }
    _fastRcnn
  }

  def setFastRcnn(value: Sequential[Float]): Unit = {
    _fastRcnn = value
  }

  protected def createFastRcnn(): Sequential[Float]

  def createRpn(): Sequential[Float]

  def createTestModel(): Sequential[Float] = {
    val model = new Sequential()
    val model1 = new ParallelTable()
    model1.add(featureAndRpnNet)
    model1.add(new Identity())
    model.add(model1)
    // connect rpn and fast-rcnn
    val middle = new ConcatTable()
    val left = new Sequential()
    val left1 = new ConcatTable()
    left1.add(selectTensor(1, 1, 1))
    left1.add(selectTensor(1, 1, 2))
    left1.add(selectTensor1(2))
    left.add(left1)
    left.add(new Proposal(preNmsTopN = param.RPN_PRE_NMS_TOP_N,
      postNmsTopN = param.RPN_POST_NMS_TOP_N, anchorParam = param.anchorParam))
    left.add(selectTensor1(1))
    // first add feature from feature net
    middle.add(selectTensor(1, 2))
    // then add rois from proposal
    middle.add(left)
    model.add(middle)
    // get the fast rcnn results and rois
    model.add(new ConcatTable().add(fastRcnn).add(selectTensor(2)))
    model
  }

  def getModel: Module[Float] = if (isTest) getTestModel else getTrainModel

  def selectTensorNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable(depth)))
    module.add(new SelectTable(depths(depths.length - 1)))
  }

  /**
   * select tensor from nested tables
   * @param depths a serious of depth to use when fetching certain tensor
   * @return a wanted tensor
   */
  def selectTensor(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length - 1).foreach(depth =>
      module.add(new SelectTable(depth)))
    module.add(new SelectTable(depths(depths.length - 1)))
  }

  def selectTensor1(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }

  def selectTensor1NoBack(depth: Int): SelectTable[Float] = {
    new SelectTable(depth)
  }

  def selectTableNoBack(depths: Int*): Sequential[Float] = {
    val module = new Sequential[Float]()
    depths.slice(0, depths.length).foreach(depth =>
      module.add(new SelectTable(depth)))
    module
  }


  def initParameters(module: Module[Float], init: (Double, Double)): Unit = {
    val params = module.parameters()
    params._1(0).apply1(_ => RNG.normal(0, init._1).toFloat)
    params._1(1).apply1(_ => init._2.toFloat)
  }

  def loadFromCaffeOrCache(dp: String, mp: String): this.type = {
    val cachedPath = mp.substring(0, mp.lastIndexOf(".")) + ".bigdl"
    val mod = FileUtil.load[(Sequential[Float], Sequential[Float])](cachedPath)
    mod match {
      case Some((featureAndRpn, fastRcnn)) =>
        logger.info(s"load model with caffe weight from cache $cachedPath")
        setFeatureAndRpnNet(featureAndRpn)
        setFastRcnn(fastRcnn)
      case _ =>
        Module.loadCaffe[Float](getModel, dp, mp, phase == TEST)
        DlFile.save((featureAndRpnNet, fastRcnn), cachedPath, isOverwrite = true)
    }
    this
  }

}

object FasterRcnn {
  def apply(modelType: ModelType, phase: PhaseType = TEST,
    caffeModel: Option[(String, String)] = None): FasterRcnn = {

    def getFasterRcnn(modelType: ModelType): FasterRcnn = {
      modelType match {
        case VGG16 =>
          new VggFRcnn(phase)
        case PVANET =>
          new PvanetFRcnn(phase)
        case _ =>
          throw new Exception("unsupport network")
      }
    }

    Random.setSeed(3)
    val fasterRcnnModel = caffeModel match {
      case Some((dp: String, mp: String)) =>
        // caffe pretrained model
        getFasterRcnn(modelType)
          .loadFromCaffeOrCache(dp, mp)
      case _ => getFasterRcnn(modelType)
    }
    fasterRcnnModel
  }
}
