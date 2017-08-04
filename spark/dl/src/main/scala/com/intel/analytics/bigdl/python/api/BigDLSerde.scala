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

// Let the package prefix with "org.apache.spark" to access classes of Spark
// Some of the code originally from PySpark
package org.apache.spark.bigdl.api.python

import java.io.OutputStream
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.python.api.{JTensor, Sample, TestResult}
import net.razorvine.pickle._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.python.SerDeUtil
import org.apache.spark.mllib.api.python.SerDe
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

/**
 * Basic SerDe utility class.
 */
private[spark] abstract class BigDLSerDeBase {

  val PYSPARK_PACKAGE: String

  def initialize(): Unit

  def dumps(obj: AnyRef): Array[Byte] = {
    obj match {
      // Pickler in Python side cannot deserialize Scala Array normally. See SPARK-12834.
      case array: Array[_] => new Pickler().dumps(array.toSeq.asJava)
      case _ => new Pickler().dumps(obj)
    }
  }

  def loads(bytes: Array[Byte]): AnyRef = {
    new Unpickler().loads(bytes)
  }

  def asTupleRDD(rdd: RDD[Array[Any]]): RDD[(Int, Int)] = {
    rdd.map(x => (x(0).asInstanceOf[Int], x(1).asInstanceOf[Int]))
  }

  def fromTuple2RDD(rdd: RDD[(Any, Any)]): RDD[Array[Any]] = {
    rdd.map(x => Array(x._1, x._2))
  }


  def javaToPython(jRDD: JavaRDD[Any]): JavaRDD[Array[Byte]] = {
    jRDD.rdd.mapPartitions { iter =>
      initialize() // let it called in executor
      new SerDeUtil.AutoBatchedPickler(iter)
    }
  }


  def pythonToJava(pyRDD: JavaRDD[Array[Byte]], batched: Boolean)
  : JavaRDD[Any] = {
    pyRDD.rdd.mapPartitions { iter =>
      initialize()
      val unpickle = new Unpickler
      iter.flatMap { row =>
        val obj = unpickle.loads(row)
        if (batched) {
          obj match {
            case list: JArrayList[_] => list.asScala
            case arr: Array[_] => arr
          }
        } else {
          Seq(obj)
        }
      }
    }.toJavaRDD()
  }
}

/**
 * SerDe utility functions for BigDL.
 */
object BigDLSerDe extends BigDLSerDeBase with Serializable {

  val PYSPARK_PACKAGE = "bigdl.util.common"
  val LATIN1 = "ISO-8859-1"


  /**
   * Base class used for pickle
   */
  private[python] abstract class BigDLBasePickler[T: ClassTag]
    extends IObjectPickler with IObjectConstructor {

    val PYSPARK_PACKAGE = "bigdl.util.common"
    val LATIN1 = "ISO-8859-1"

    private val cls = implicitly[ClassTag[T]].runtimeClass
    println("cls.getname: " + cls.getName)
    private val module = PYSPARK_PACKAGE
    private val name = cls.getSimpleName

    def register(): Unit = {
      Pickler.registerCustomPickler(this.getClass, this)
      Pickler.registerCustomPickler(cls, this)
      Unpickler.registerConstructor(module, name, this)
      println(s"BigDLBasePickler registering: $module  $name")
    }

    def pickle(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      if (obj == this) {
        out.write(Opcodes.GLOBAL)
        out.write((module + "\n" + name + "\n").getBytes)
      } else {
        pickler.save(this) // it will be memorized by Pickler
        saveState(obj, out, pickler)
        out.write(Opcodes.REDUCE)
      }
    }

    private[python] def saveObjects(out: OutputStream, pickler: Pickler, objects: Any*) = {
      if (objects.length == 0 || objects.length > 3) {
        out.write(Opcodes.MARK)
      }
      objects.foreach(pickler.save)
      val code = objects.length match {
        case 1 => Opcodes.TUPLE1
        case 2 => Opcodes.TUPLE2
        case 3 => Opcodes.TUPLE3
        case _ => Opcodes.TUPLE
      }
      out.write(code)
    }

    protected def getBytes(obj: Object): Array[Byte] = {
      if (obj.getClass.isArray) {
        obj.asInstanceOf[Array[Byte]]
      } else {
        obj.asInstanceOf[String].getBytes(LATIN1)
      }
    }

    private[python] def saveState(obj: Object, out: OutputStream, pickler: Pickler)
  }

  private[python] class SamplePickler extends BigDLBasePickler[Sample] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val record = obj.asInstanceOf[Sample]
      saveObjects(out,
        pickler,
        record.features,
        record.label, record.featuresShape, record.labelShape)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 5) {
        throw new PickleException("should be 5, not : " + args.length)
      }
      new Sample(args(0).asInstanceOf[JArrayList[Any]],
        args(1).asInstanceOf[JArrayList[Any]],
        args(2).asInstanceOf[JArrayList[Int]],
        args(3).asInstanceOf[JArrayList[Int]],
        args(4).asInstanceOf[String])
    }
  }

  private[python] class TestResultPickler extends BigDLBasePickler[TestResult] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val testResult = obj.asInstanceOf[TestResult]
      saveObjects(out,
        pickler,
        testResult.result,
        testResult.totalNum, testResult.method)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3, not : " + args.length)
      }
      new TestResult(args(0).asInstanceOf[Float],
        args(1).asInstanceOf[Int],
        args(2).asInstanceOf[String])
    }
  }

  private[python] class JTensorPickler extends BigDLBasePickler[JTensor] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val testResult = obj.asInstanceOf[JTensor]
      saveObjects(out,
        pickler,
        testResult.storage,
        testResult.shape, testResult.bigdlType)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3, not : " + args.length)
      }
      val bigdl_type = args(2).asInstanceOf[String]
      // Only allow float and double for now same as Tensor
      val rawStorage = args(0).asInstanceOf[JArrayList[Double]].asScala
      val storage = bigdl_type match {
        case "float" =>
          rawStorage.map(_.toFloat).toList.asJava
        case "double" =>
          rawStorage.toList.asJava
        case _ => throw new IllegalArgumentException("Only support float and double for now")
      }
      new JTensor(storage.asInstanceOf[JList[Any]],
        args(1).asInstanceOf[JArrayList[Int]],
        bigdl_type)
    }
  }

  var initialized = false

  override def initialize(): Unit = {
    synchronized {
      if (!initialized) {
        SerDe.initialize()
        new SamplePickler().register()
        new TestResultPickler().register()
        new JTensorPickler().register()
        initialized = true
      }
    }
  }
  // will not called in Executor automatically
  initialize()
}
