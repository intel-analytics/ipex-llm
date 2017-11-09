/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.utils.tf.loaders

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.{T, Table, serializer}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.runtime.universe
import scala.reflect.ClassTag

/**
 * Sometimes parameters pass as tensor from some previous nodes in graph. So we can't construct
 * module at module define time. In such case, you can use the AdapterOperation to wrapper the
 * module and create it in runtime.
 *
 * Please note you must guarantee the input parameter won't change each time.
 * @param configIndexes configuration tensor indexes, start from 1 and -1 specify the last one
 * @param build build function
 * @tparam T Numeric type. Only support float/double now
 */
class Adapter[T: ClassTag](
  val configIndexes: Array[Int],
  val build: Array[Tensor[_]] => AbstractModule[Activity, Activity, T]
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Activity, T]{

  private var module : AbstractModule[Activity, Activity, T] = _
  private var indexes : Array[Int] = _
  private var dataIndexes: Array[Int] = _
  private var zeroGrads: Array[Tensor[_]] = _
  private var realInput: Activity = _
  private var initTensors: Array[Tensor[_]] = _

  override def updateOutput(input: Table): Activity = {
    if (module == null) {
      val l = input.length()
      indexes = configIndexes.map(getPositiveIndex(_, l))
      val tensors = indexes.map(i => input[Tensor[_]](i))
      initTensors = tensors.map(_.clone())
      dataIndexes = getDataIndexes(indexes, l)
      zeroGrads = tensors.map(t => t.emptyInstance().resizeAs(t))
      module = build(tensors)
    } else {
      indexes.map(i => input[Tensor[_]](i)).zip(initTensors).foreach(tensors => {
        require(tensors._1 == tensors._2, s"constant tensor is changed. " +
          s"\noriginal\n${tensors._2}\nnow\n${tensors._1}")
      })
    }

    realInput = if (dataIndexes.length == 1) {
      input[Tensor[_]](dataIndexes(0))
    } else {
      val t = T()
      dataIndexes.map(i => t.insert(input[Tensor[_]](i)))
      t
    }

    output = module.forward(realInput)
    output
  }

  private def getPositiveIndex(index: Int, length: Int): Int = {
    if (index > 0) index else length + index + 1
  }

  private def getDataIndexes(indexs: Array[Int], length: Int): Array[Int] = {
    (1 to length).filterNot(indexs.contains(_)).toArray
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    val realGradInput = module.updateGradInput(realInput, gradOutput)
    gradInput = T()
    var i = 0
    while(i < indexes.length) {
      gradInput(indexes(i)) = zeroGrads(i)
      i += 1
    }
    if (dataIndexes.length == 1) {
      gradInput(dataIndexes.head) = realGradInput
    } else {
      i = 0
      while (i < dataIndexes.length) {
        gradInput(dataIndexes(i)) = realGradInput.toTable.apply[Activity](i + 1)
        i += 1
      }
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Activity): Unit = {
    module.accGradParameters(realInput, gradOutput)
  }
}

object Adapter extends ModuleSerializable {
  def apply[T: ClassTag](
    configIndexes: Array[Int], build: Array[Tensor[_]] => AbstractModule[Activity, Activity, T]
  )(implicit ev: TensorNumeric[T]): Adapter[T] = {
    new Adapter(configIndexes, build)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val configIndexes = DataConverter.getAttributeValue(context, attrMap.get("configIndexes")).
      asInstanceOf[Array[Int]]

    val adapter = Adapter[T](configIndexes, null)


    adapter.module = DataConverter.getAttributeValue(context, attrMap.get("module")).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    adapter.initTensors = DataConverter.getAttributeValue(context, attrMap.get("tensors")).
      asInstanceOf[Array[Tensor[_]]]

    adapter.indexes = DataConverter.getAttributeValue(context, attrMap.get("indexes")).
      asInstanceOf[Array[Int]]

    adapter.dataIndexes = DataConverter.getAttributeValue(context, attrMap.get("dataIndexes")).
      asInstanceOf[Array[Int]]

    adapter.zeroGrads = adapter.initTensors.map(t => t.emptyInstance().resizeAs(t))

    adapter
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val adapterModule = context.moduleData.module.asInstanceOf[Adapter[T]]

    val moduleBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, moduleBuilder, adapterModule.module,
      serializer.ModuleSerializer.abstractModuleType)
    bigDLModelBuilder.putAttr("module", moduleBuilder.build)

    val configIndexesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, configIndexesBuilder, adapterModule.configIndexes,
      universe.typeOf[Array[Int]])
    bigDLModelBuilder.putAttr("configIndexes", moduleBuilder.build)

    val indexesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, indexesBuilder, adapterModule.indexes,
      universe.typeOf[Array[Int]])
    bigDLModelBuilder.putAttr("indexes", indexesBuilder.build)

    val dataIndexesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, dataIndexesBuilder, adapterModule.dataIndexes,
      universe.typeOf[Array[Int]])
    bigDLModelBuilder.putAttr("dataIndexes", dataIndexesBuilder.build)

    val tensorsBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, tensorsBuilder, adapterModule.initTensors,
      universe.typeOf[Array[Tensor[_]]])
    bigDLModelBuilder.putAttr("tensors", tensorsBuilder.build)

    bigDLModelBuilder.putAttr("configIndexes", moduleBuilder.build)

  }
}
