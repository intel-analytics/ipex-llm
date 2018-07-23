
package com.intel.analytics.bigdl

import java.util.concurrent.{Callable, Executors, Future}

import com.intel.analytics.bigdl.models.resnet.{ImageNetDataSet, ResNet}
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.optim.DistriOptimizer.getClass
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}


object Test {
  def main(args: Array[String]): Unit = {
  /*
    val curModel =
      ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
        "optnet" -> true, "dataSet" -> DatasetType.ImageNet))

    val parameters = curModel.getParameters()._2

    var len = parameters.storage().array().length
*/
  val logger: Logger = Logger.getLogger(getClass)

  val conf = Engine.createSparkConf().setAppName("Train ResNet on ImageNet2012")
    .set("spark.rpc.message.maxSize", "200")
    val sc = new SparkContext(conf)
    Engine.init

    val nodes = Engine.nodeNumber

    val cores = Engine.coreNumber

    println(s"total nodes : ${nodes}")

    println(s"cores per node : ${cores}")

    logger.info(s"total nodes : ${nodes}")

    logger.info(s"cores per node : ${cores}")

    val totalLen = 26214400

    val splits = args(0).toInt

    val comms = sc.parallelize((0 until nodes), nodes).mapPartitions(p => {
      val partitionId = TaskContext.getPartitionId()
      val comm = new AsyncComm(splits, nodes, partitionId)
      val data = Tensor[Float](1, totalLen / nodes).rand()
      Iterator.single(Cache(data, comm))
    }).persist
    var total : Float = 0L
    (0 until 10).map( i => {
      println(s"execut ${i} times")
      val times = comms.mapPartitions(p => {
        val start = System.nanoTime
        val cache = p.next
        cache.asyncComm.sync(cache.data)
        val end = System.nanoTime
        Iterator.single(((end - start).toFloat /1e9).toFloat)
      }).collect
      times.foreach(time => total += time)
    })
    val avg = total.toFloat / (10 * nodes)
    logger.info(s"total nodes : ${nodes}")
    logger.info(s"cores per node : ${cores}")
    logger.info(s"Split is : ${splits}")
    logger.info(s"Average time is : ${avg}")
  }
}

case class Cache(data: Tensor[Float], asyncComm: AsyncComm)

class AsyncComm(splits: Int, partition: Int, partitionId: Int) extends Serializable {
  val executorService = Executors.newFixedThreadPool(10)
  def sync(input: Tensor[Float]): Unit = {
    val tsize = input.nElement()
    val step = tsize / splits
    var futures = new Array[Future[Tensor[Float]]](splits)
    var i: Int = 0
    while (i < splits) {
      val viewTensor = Tensor(input.storage(), tsize - step + 1)
      val splitTensor = Tensor[Float]().resizeAs(viewTensor).copy(viewTensor )
      futures(i) = executorService.submit(new SyncTask(splitTensor, partition, i))
      i += 1
    }
    futures.foreach(f => f.get())
  }

  class SyncTask(input: Tensor[Float], partition: Int, split: Int) extends Callable[Tensor[Float]] {
    override def call(): Tensor[Float] = {
      val curr = System.currentTimeMillis()
      BlockManagerWrapper.putSingle(getBlockId("SyncTask" + split + partitionId),
        input, StorageLevel.MEMORY_ONLY_SER)
     (0 until partition).map(index => {
       println(s"get ${index} time")
       BlockManagerWrapper.getLocalOrRemoteBytes(getBlockId("SyncTask" + split + index)).get
     })
      input
    }

    private def getBlockId(name: String): BlockId = {
      SparkExtension.getLocalBlockId(name)
    }
  }
}

class ParameterService(size: Int, partitionNum: Int) extends Serializable {
  @transient private var taskSize = 0
  @transient private var extraSize = 0
  @transient private var partitionId: Int = 0

  /** Tensor to hold a slice of the global weights. */
  @transient lazy val weightPartition: Tensor[Float] = readWeightPartition()

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    in.defaultReadObject()
    taskSize = size / partitionNum
    extraSize = size % partitionNum
    partitionId = TaskContext.getPartitionId()
  }

  private def readWeightPartition(): Tensor[Float] = {
    val blockId = getWeightPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[Float]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  private def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId("weights" + partitionId)
  }
}
