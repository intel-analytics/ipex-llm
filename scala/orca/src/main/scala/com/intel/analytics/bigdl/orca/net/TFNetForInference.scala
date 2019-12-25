/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.tfpark.{GraphRunner, TFUtils}
import org.slf4j.LoggerFactory
import org.tensorflow.framework.{GraphDef, MetaGraphDef}
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.{DataType, SavedModelBundle}

import scala.reflect.ClassTag


private[zoo] class TFNetForInference(graphRunner: GraphRunner,
                                    inputs: Array[String],
                                    inputTypes: Array[Int],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    variableTypes: Array[Int],
                                    variableAssignPlaceholders: Array[String],
                                    assignVariableOps: Array[String],
                                    initWeights: Array[Tensor[Float]],
                                    initOp: Option[String])
  extends AbstractModule[Activity, Activity, Float] with Predictable[Float] {

  protected val module: Module[Float] = this
  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }

  private val weights = initWeights

  private val gradWeights = variables.map(_ => Tensor[Float]())

  private val graphOutputs = {
    val graphOuts = Vector.newBuilder[Tensor[Float]]

    var i = 0
    while (i < outputs.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    graphOuts.result()
  }

  output = {
    if (outputs.length == 1) {
      graphOutputs(0)
    } else {
      val out = T()
      var i = 0
      while (i < outputs.length) {
        out.insert(graphOutputs(i))
        i += 1
      }
      out
    }
  }

  gradInput = {
    if (inputs.length == 1) {
      Tensor[Float]()
    } else {
      val t = T()
      var i = 0
      while (i < inputs.length) {
        t.insert(Tensor[Float]())
        i = i + 1
      }
      t
    }
  }

  private def runInitOp(): Unit = {
    graphRunner.run(
      input = Vector.empty,
      inputTypes = Vector.empty,
      output = Vector.empty,
      inputNames = Vector.empty,
      outputNames = Vector.empty,
      targets = Vector(initOp.get)
    )
  }

  private def setVariableIntoTF(weights: Array[Tensor[Float]],
                                inputNames: Array[String],
                                variableTypes: Array[DataType],
                                assignOps: Array[String]) = {
    graphRunner.run(
      input = weights.toVector,
      inputTypes = variableTypes.toVector,
      output = Vector.empty,
      inputNames = inputNames.toVector,
      outputNames = Vector.empty,
      targets = assignOps.toVector
    )
  }

  @transient
  private lazy val variableInited = {
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(TFUtils.tfenum2datatype), assignVariableOps)
    true
  }

  @transient
  private lazy val tableInited = {
    if (initOp.isDefined) {
      runInitOp()
    }
    true
  }

  override def updateOutput(input: Activity): Activity = {
    Utils.timeIt("updateOutput") {

      assert(variableInited)
      assert(tableInited)

      val feeds = Utils.activity2VectorBuilder(input)

      val types = inputTypes.toVector.map(TFUtils.tfenum2datatype)

      graphRunner.run(
        input = feeds.result(),
        inputTypes = types,
        output = graphOutputs,
        inputNames = inputs.toVector,
        outputNames = outputs.toVector,
        targets = Vector.empty)
    }

    output
  }

  override def updateGradInput(
           input: Activity,
           gradOutput: Activity): Activity = {
    NetUtils.generateZeroGrad(input, gradInput)
    gradInput
  }
}

object TFNetForInference {

  TFNet

  private val INIT_OP_SIGNATURE_KEY = "__saved_model_init_op"
  private val MAIN_OP_KEY = "saved_model_main_op"
  private val LEGACY_INIT_OP_KEY = "legacy_init_op"
  private val DEFAULT_TAG = "serve"
  private val DEFAULT_SIGNATURE = "serving_default"


  val logger = LoggerFactory.getLogger(getClass)

  import scala.collection.JavaConverters._

  val frameworkDataType2Class = Map(
    org.tensorflow.framework.DataType.DT_FLOAT -> classOf[java.lang.Float],
    org.tensorflow.framework.DataType.DT_INT32 -> classOf[java.lang.Integer],
    org.tensorflow.framework.DataType.DT_INT64 -> classOf[java.lang.Long]
  )

  val frameworkDataType2DataType = Map(
    org.tensorflow.framework.DataType.DT_FLOAT -> org.tensorflow.DataType.FLOAT,
    org.tensorflow.framework.DataType.DT_INT32 -> org.tensorflow.DataType.INT32,
    org.tensorflow.framework.DataType.DT_INT64 -> org.tensorflow.DataType.INT64
  )

  /*
  load TensorFlow's saved_model

  TensorFlow's Java API provides a SavedModelBundle function that will
  return a graph and a session. However, we cannot use the graph and
  session in TFNet as both of them are not serializable, thus cannot
  be broadcast to executors.

  To solve this problem, the follow approach is implemented:
  Step 1. Find all the variables in the graph and get those variable
          values out using the returned session
  Step 2. Pass those variables to TFNet as a model's weights
  Step 3. Export the returned graph as byte array and pass to TFNet
  Step 4. Create a new graph and session on executor
  Step 5. Initialize the variables on executor using model's weights

  For enable reading and re-assigning variables values, additional
  operations need to be added to the graph. For each variable, a read
  operation (for resource variable), a assign operation and a placeholder
  along with the assign operation are added to the graph.

   */
  def fromSavedModel(modelPath: String, tag: Option[String],
                     signature: Option[String],
                     inputs: Option[Array[String]],
                     outputs: Option[Array[String]],
                     sessionConfig: Array[Byte]): TFNetForInference = {

    if (tag.isEmpty) {
      logger.warn(s"Loading TensorFlow SavedModel: " +
        s"SavedModel tag is not defined, using <$DEFAULT_TAG>")
    }
    val savedModelBundle = SavedModelBundle.load(modelPath, tag.getOrElse(DEFAULT_TAG))

    val metaGraphDef = MetaGraphDef.parseFrom(savedModelBundle.metaGraphDef())
    val initOp = getInitOp(metaGraphDef)

    val (inputNames, outputNames) = if (!(inputs.isDefined && outputs.isDefined)) {
      if (signature.isEmpty) {
        logger.warn("Loading TensorFlow SavedModel: SavedModel signature is not defined," +
          s"using <$DEFAULT_SIGNATURE>")
      }
      getInputOutputNames(metaGraphDef, signature.getOrElse(DEFAULT_SIGNATURE))
    } else {
      if (inputs.isEmpty || outputs.isEmpty) {
        throw new IllegalArgumentException("inputs and outputs tensor names must be defined" +
          " if signature is not defined")
      }
      (inputs.get, outputs.get)
    }

    val graph = savedModelBundle.graph()
    val ops = Ops.create(graph).withSubScope("analytics-zoo")

    val variableTypes = Set("Variable", "VariableV2", "VarHandleOp")
    val graphBytes = graph.toGraphDef

    val graphDef = GraphDef.parseFrom(graphBytes)

    // the following map function add a read operation, an assign operation
    // and a placeholder for each variable in the graph
    val newOps = graphDef.getNodeList.asScala.filter{ node =>
      variableTypes(node.getOp)
    }.map{ x =>
      val name = x.getName
      val dataType = x.getAttrMap.get("dtype").getType
      val opType = x.getOp
      val operation = graph.operation(name)
      val dataTypeClass = frameworkDataType2Class(dataType)
      val operationOutput = operation.output(0)
      if (opType == "VarHandleOp") {
        val readVariable = ops.readVariableOp(operationOutput, dataTypeClass)
        val floatVariable = ops.dtypes.cast(readVariable, classOf[java.lang.Float])
        val placeholder = ops.placeholder(dataTypeClass,
          Placeholder.shape(readVariable.asOutput().shape()))

        // do it manually to get a reference of the op and get the op name
        val builder = graph.opBuilder("AssignVariableOp",
          ops.scope().makeOpName("AssignVariableOp"))
        builder.addInput(operationOutput)
        builder.addInput(placeholder.asOutput())
        val assignOp = builder.build()
        (floatVariable.asOutput().op().name(),
          placeholder.asOutput().op().name(), assignOp.name(),
          dataType, operationOutput.shape(), operation.name())
      } else {
        val readVariable = operationOutput
        val floatVariable = ops.dtypes.cast(readVariable, classOf[java.lang.Float])
        val placeholder = ops.placeholder(dataTypeClass,
          Placeholder.shape(operationOutput.shape()))

        // do it manually to get a reference of the op and get the op name
        val builder = graph.opBuilder("Assign",
          ops.scope().makeOpName("Assign"))
        builder.addInput(operationOutput)
        builder.addInput(placeholder.asOutput())
        val assignOp = builder.build()
        (floatVariable.asOutput().op().name(),
          placeholder.asOutput().op().name(), assignOp.name(),
          dataType, operationOutput.shape(), operation.name())
      }
    }

    val readVariableNames = newOps.map(_._1)
    val placeholderNames = newOps.map(_._2)
    val assign = newOps.map(_._3)
    val dataTypes = newOps.map(_._4)
    val dataShapes = newOps.map(x => (x._5, x._6))

    val graphdef = GraphDef.parseFrom(graph.toGraphDef)

    val graphRunner = new GraphRunner(
      graph.toGraphDef,
      null, null, null, null,
      TFNet.defaultSessionConfig.toByteArray())

    val session = savedModelBundle.session()

    // The ideal approach would be fetching all the variables all at once, but
    // some of the variable might be not be in the saved_model, thus the variable
    // is not initialized and cannot be fetched. Currently, there is no way to know
    // which one is not initialized until we fetch it and get an exception.
    val weights = readVariableNames.zip(dataShapes).map { case (name, (shape, originalName)) =>
      val runner = session.runner()
      runner.fetch(name)
      try {
        val value = runner.run()
        val bigdlTensor = Tensor[Float]()
        TFUtils.tf2bigdl(value.get(0), bigdlTensor)
        value.get(0).close()
        bigdlTensor
      } catch {
        case _: Exception =>
          TFNetForInference.logger.warn(s"Cannot find variable value for <$originalName>, " +
            s"using default value zero")
          val shapeArr = new Array[Int](shape.numDimensions())
          var i = 0
          while (i < shape.numDimensions()) {
            shapeArr(i) = shape.size(i).toInt
            i += 1
          }
          Tensor[Float](sizes = shapeArr)
      }
    }.toArray

    val inputTypes = inputNames.map { name =>
      val opAndPort = name.split(":")
      val op = opAndPort.head
      val port = opAndPort(1)
      val opRef = graph.operation(op)
      if (opRef == null) {
        throw new IllegalArgumentException(s"Cannot find input op <$name>")
      }
      TFUtils.tfdatatype2enum(opRef.output(port.toInt).dataType())
    }



    // clean up native resources
    savedModelBundle.close()

    new TFNetForInference(graphRunner = graphRunner,
      inputs = inputNames,
      inputTypes = inputTypes,
      outputs = outputNames,
      variables = readVariableNames.toArray,
      variableTypes = dataTypes.map(_.getNumber).toArray,
      variableAssignPlaceholders = placeholderNames.toArray,
      assignVariableOps = assign.toArray,
      initWeights = weights, initOp)
  }

  def getInputOutputNames(metaGraphDef: MetaGraphDef,
                          signature: String): (Array[String], Array[String]) = {
    val signatureDef = metaGraphDef.getSignatureDefOrThrow(signature)
    val inputMap = signatureDef.getInputsMap
    val sortedInputKeys = inputMap.keySet().asScala.toArray.sorted
    val inputNames = sortedInputKeys.map(inputMap.get(_).getName)
    val outputMap = signatureDef.getOutputsMap
    val sortedOutputKeys = outputMap.keySet().asScala.toArray.sorted
    val outputNames = sortedOutputKeys.map(outputMap.get(_).getName)
    (inputNames, outputNames)
  }


  def getOpFromSignatureDef(metaGraphDef: MetaGraphDef, signatureKey: String): Option[String] = {
    if (metaGraphDef.containsSignatureDef(signatureKey)) {
      val signatureDef = metaGraphDef.getSignatureDefOrThrow(signatureKey)
      val tensorInfo = signatureDef.getOutputsOrThrow(signatureKey)
      Some(tensorInfo.getName)
    } else {
      None
    }
  }

  def getOpFromCollection(metaGraphDef: MetaGraphDef, opKey: String): Option[String] = {
    if (metaGraphDef.containsCollectionDef(opKey)) {
      val collectionDef = metaGraphDef.getCollectionDefOrThrow(opKey)
      val name = collectionDef.getNodeList.getValue(0)
      Some(name)
    } else {
      None
    }
  }

  def getInitOp(metaGraphDef: MetaGraphDef): Option[String] = {
    val signatureOp = getOpFromSignatureDef(metaGraphDef, INIT_OP_SIGNATURE_KEY)
    val mainOp = getOpFromCollection(metaGraphDef, MAIN_OP_KEY)
    val legacyIntOP = getOpFromCollection(metaGraphDef, LEGACY_INIT_OP_KEY)

    val result = List(signatureOp, mainOp, legacyIntOP).flatten
    result.headOption
  }
}


