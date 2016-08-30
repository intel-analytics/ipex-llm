package com.intel.webscaleml.nn.optim

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{torch, Tensor}
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * Represent the distributed train data
  *
  * @tparam D raw data type
  * @tparam T numeric type of the tensor to be generated
  */
trait DataSet[D, T] extends Serializable {

  /**
    * Fetch several batches of data for training. Each batch is made up of some stacks. Each stack has two tensors,
    * one is input data and the other is the labels
    *
    * @return
    */
  def fetch() : RDD[Iterator[(Tensor[T], Tensor[T])]]

  /**
    * Total count of the data
    *
    * @return
    */
  def total() : Long

  def getSparkContext() : SparkContext

  def getPartitionNum() : Int

  def fetchAll() : RDD[(Tensor[T], Tensor[T])]

  def partitions() : RDD[_]
}

/**
  * The data set support epoch. A epoch mean go through all data
  */
trait HasEpoch {
  /**
    * shuffle the data
    */
  def reset() : Unit

  /**
    * is current epoch done
 *
    * @return
    */
  def epochFinished() : Boolean
}

/**
  * Fetch some batch of data by sampling the data
 *
  * @param dataSets raw data
  * @param toTensor construct tensor from the raw data
  * @param stackSize stack size
  * @param batchSize batch size
  * @param batchNum how many batches in each fetch
  * @tparam D raw data type
  * @tparam T numeric type of the tensor to be generated
  */
class SampledBatchDataSet[D : ClassTag, @specialized(Float, Double) T : ClassTag](
    val dataSets: RDD[D],
    val toTensor: (Seq[D], Tensor[T], Tensor[T]) => (Tensor[T], Tensor[T]),
    val stackSize : Int,
    val batchSize : Int,
    val batchNum : Int)(implicit ev: TensorNumeric[T])
  extends DataSet[D, T] {

  lazy val count = dataSets.count()
  lazy val partitionNum = dataSets.partitions.length

  override def fetch(): RDD[Iterator[(Tensor[T], Tensor[T])]] = {
    dataSets.sample(false, batchNum * batchSize * partitionNum / count.toDouble, System.nanoTime())
      .mapPartitions(iter => {
        val input = torch.Tensor[T]()
        val target = torch.Tensor[T]()
        iter.grouped(batchSize).map(_.grouped(stackSize).map(toTensor(_, input, target)))
      })
  }

  override def total(): Long = count

  override def getSparkContext(): SparkContext = dataSets.sparkContext

  override def getPartitionNum(): Int = dataSets.partitions.length

  override def fetchAll(): RDD[(Tensor[T], Tensor[T])] = {
    dataSets.mapPartitions(iter => {
        val input = torch.Tensor[T]()
        val target = torch.Tensor[T]()
        iter.grouped(batchSize).map(toTensor(_, input, target))
      })
  }

  override def partitions(): RDD[_] = dataSets
}

/**
  * Fetch data sequentially from data set. The sequence can be shuffled.
 *
  * @param dataSets raw data
  * @param toTensor construct tensor from the raw data
  * @param stackSize stack size
  * @param batchSize batch size
  * @param batchNum how many batches in each fetch
  * @tparam D raw data type
  * @tparam T numeric type of the tensor to be generated
  */
class ShuffleBatchDataSet[D : ClassTag, @specialized(Float, Double) T : ClassTag](
    val dataSets: RDD[D],
    val toTensor: (Seq[D], Tensor[T], Tensor[T]) => (Tensor[T], Tensor[T]),
    val stackSize : Int,
    val batchSize : Int, // use stackSize
    val batchNum : Int = 1) (implicit ev: TensorNumeric[T])
  extends DataSet[D, T] with HasEpoch {

  require(stackSize <= batchSize && stackSize > 0)
  require(batchSize > 0 && batchNum > 0)

  private var curPosition = 0

  private var shuffledIndex : RDD[Array[Int]] = dataSets.mapPartitions(iter => {
    Iterator.single(Array.range(0, iter.length))
  }).setName("Shuffled Index").cache()
  shuffledIndex.count()

  lazy private val maxLength = shuffledIndex.map(_.length).max()
  lazy private val count = shuffledIndex.map(_.length).sum().toLong


  override def fetch(): RDD[Iterator[(Tensor[T], Tensor[T])]] = {
    val position = curPosition
    val result = dataSets.zipPartitions(shuffledIndex)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexBuffer = new Array[Int](batchNum * batchSize)
      var n = 0
      while(n < batchNum * batchSize) {
        indexBuffer(n) = indexes((position + n) % indexes.length)
        n += 1
      }

      val orderedIndex = indexBuffer.sortWith(_ < _)
      val dataBuffer = new Array[D](batchNum * batchSize)
      n = 0
      var i = 0
      while(dataIter.hasNext && n < orderedIndex.length) {
        val d = dataIter.next()
        if(i == orderedIndex(n)) {
          dataBuffer(n) = d
          n += 1
        }
        i += 1
      }
      require(n == batchNum * batchSize)

      val input = torch.Tensor[T]()
      val target = torch.Tensor[T]()
      dataBuffer.grouped(batchSize).map(_.grouped(stackSize).map(toTensor(_, input, target)))
    })
    curPosition += batchNum * batchSize
    result
  }

  override def reset(): Unit = {
    shuffledIndex.unpersist()
    shuffledIndex = dataSets.mapPartitions(iter => {
      Iterator.single(Array.range(0, iter.length))
    }).map(Utils.inPlaceShuffle(_)).setName("Shuffled Index").cache()
    curPosition = 0
  }

  override def epochFinished(): Boolean = {
    curPosition >= maxLength
  }

  override def total(): Long = count

  override def getSparkContext(): SparkContext = dataSets.sparkContext

  override def getPartitionNum(): Int = dataSets.partitions.length

  override def fetchAll(): RDD[(Tensor[T], Tensor[T])] = {
    dataSets.mapPartitions(iter => {
      val input = torch.Tensor[T]()
      val target = torch.Tensor[T]()
      iter.grouped(batchSize).map(toTensor(_, input, target))
    })
  }

  override def partitions(): RDD[_] = dataSets
}

/**
  * Fetch whole data as a sequence of batches. One fetch can go through data for several times. The sequence of data can
  * be shuffled
 *
  * @param dataSets raw data
  * @param toTensor construct tensor from the raw data
  * @param stackSize stack size
  * @param batchSize batch size
  * @param innerLoop how much loops go through the whole dataset
  * @tparam D raw data type
  * @tparam T numeric type of the tensor to be generated
  */
class ShuffleFullBatchDataSet[D : ClassTag, @specialized(Float, Double) T : ClassTag](
    val dataSets: RDD[D],
    val toTensor: (Seq[D], Tensor[T], Tensor[T]) => (Tensor[T], Tensor[T]),
    val stackSize : Int, val batchSize : Int, val innerLoop : Int) (implicit ev: TensorNumeric[T])
  extends DataSet[D, T] with HasEpoch {

  require(stackSize <= batchSize && stackSize > 0)
  require(batchSize > 0 && innerLoop > 0)

  private var isFinished = false

  private val cacheData : RDD[Array[D]] = dataSets.mapPartitions(iter => {
    Iterator.single(iter.foldLeft(new ArrayBuffer[D]())((b, d) => b += d).toArray)
  }).setName("Cached Data").cache()

  private var shuffedIndex : RDD[Array[Int]] = cacheData.mapPartitions(iter => {
    Iterator.single(Array.range(0, iter.next().length))
  }).setName("Shuffled Index").cache()

  lazy private val count = shuffedIndex.map(_.length).sum().toLong

  override def fetch(): RDD[Iterator[(Tensor[T], Tensor[T])]] = {
    val result = cacheData.zipPartitions(shuffedIndex)((blockIter, indexIter) => {
      val data = blockIter.next()
      val indexes = indexIter.next()

      val input = torch.Tensor[T]()
      val target = torch.Tensor[T]()
      val batchNum = math.ceil(data.length.toDouble / batchSize).toInt
      println(s"data.length ${data.length} batchSize $batchSize batchNum $batchNum stackSize $stackSize innerLoop $innerLoop")
      Iterator.range(0, batchNum * innerLoop).map( i => new ShuffleIterator(data, indexes,
        (i % batchNum) * batchSize, batchSize).grouped(stackSize).map(toTensor(_, input, target)))
    })
    isFinished = true
    result
  }

  override def reset(): Unit = {
    val tmp = shuffedIndex.map(Utils.inPlaceShuffle(_)).cache().setName("Shuffled Index")
    tmp.count()
    shuffedIndex.unpersist()
    shuffedIndex = tmp
    isFinished = false
  }

  override def epochFinished(): Boolean = isFinished

  override def total(): Long = count

  override def getSparkContext(): SparkContext = dataSets.sparkContext

  override def getPartitionNum(): Int = dataSets.partitions.length

  override def fetchAll(): RDD[(Tensor[T], Tensor[T])] = {
    dataSets.mapPartitions(iter => {
      val input = torch.Tensor[T]()
      val target = torch.Tensor[T]()
      iter.grouped(batchSize).map(toTensor(_, input, target))
    })
  }

  override def partitions(): RDD[_] = dataSets
}

class ShuffleIterator[D](
    val data : Array[D],
    val index : Array[Int],
    val startPosition : Int,
    val batchSize : Int
  ) extends Iterator[D] {

  private var count = 0

  override def hasNext: Boolean = count < batchSize

  override def next(): D = {
    val i = index((startPosition + count) % data.length)
    count += 1
    data(i)
  }
}