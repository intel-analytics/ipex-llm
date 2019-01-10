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

package com.intel.analytics.bigdl.dataset

import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.image.{LabeledBGRImage, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import org.apache.hadoop.io.Text
import org.apache.log4j.Logger
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect._

/**
 * A set of data which is used in the model optimization process. The dataset can be access in
 * a random data sample sequence. In the training process, the data sequence is a looped endless
 * sequence. While in the validation process, the data sequence is a limited length sequence.
 * User can use the data() method to get the data sequence.
 *
 * The sequence of the data is not fixed. It can be changed by the shuffle() method.
 *
 * User can create a dataset from a RDD, an array and a folder, etc. The DataSet object provides
 * many factory methods.
 *
 * @tparam D Data type
 * @tparam DataSequence Represent a sequence of data
 */
trait AbstractDataSet[D, DataSequence] {
  /**
   * Get a sequence of data
   *
   * @param train if the data is used in train. If yes, the data sequence is a looped endless
   *              sequence, or it has a limited length.
   * @return data sequence
   */
  def data(train: Boolean): DataSequence

  /**
   * Change the order of the data sequence from the data set
   */
  def shuffle(): Unit

  /**
   * Total size of the data set
   * @return
   */
  def size(): Long

  /**
   * Helper function to transform the data type in the data set.
   * @param transformer
   * @tparam C
   * @return
   */
  def transform[C: ClassTag](transformer: Transformer[D, C]): DataSet[C]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * Helper function to transform the data type in the data set.
   *
   * @param transformer
   * @tparam C
   * @return
   */
  def -> [C: ClassTag](transformer: Transformer[D, C]): DataSet[C] = {
    this.transform(transformer)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Convert current DataSet to a local DataSet, in which we use an iterator to represent the
   * data sequence.
   * @return
   */
  def toLocal(): LocalDataSet[D] = this.asInstanceOf[LocalDataSet[D]]

  /**
   * Convert current DataSet to a distributed DataSet, in which we use a RDD to represent the
   * data sequence.
   * @return
   */
  def toDistributed(): DistributedDataSet[D] = this.asInstanceOf[DistributedDataSet[D]]

  /**
   * Split current DataSet to multi parts.
   * @return A Seq(DataSet)
   */
  def getSplits(): Seq[AbstractDataSet[D, DataSequence]] = Seq(this)
}

/**
 * Manage some 'local' data, e.g. data in files or memory. We use iterator to go through the data.
 * @tparam T
 */
trait LocalDataSet[T] extends AbstractDataSet[T, Iterator[T]] {
  override def transform[C: ClassTag](transformer: Transformer[T, C]): DataSet[C] = {
    val preDataSet = this
    new LocalDataSet[C] {
      override def shuffle(): Unit = preDataSet.shuffle

      override def size(): Long = preDataSet.size()

      override def data(train: Boolean): Iterator[C] = transformer(preDataSet.data(train))
    }
  }
}

/**
 * Wrap an array as a DataSet.
 * @param buffer
 * @tparam T
 */
class LocalArrayDataSet[T] private[dataset](buffer: Array[T]) extends LocalDataSet[T] {
  override def shuffle(): Unit = {
    RandomGenerator.shuffle(buffer)
  }

  override def data(train: Boolean): Iterator[T] = {
    new Iterator[T] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (train) {
          true
        } else {
          index.get() < buffer.length
        }
      }

      override def next(): T = {
        val curIndex = index.getAndIncrement()
        if (train || curIndex < buffer.length) {
          buffer(if (train) (curIndex % buffer.length) else curIndex)
        } else {
          null.asInstanceOf[T]
        }
      }
    }
  }

  override def size(): Long = buffer.length
}

/**
 * Represent a distributed data. Use RDD to go through all data.
 *
 * @tparam T
 */
trait DistributedDataSet[T] extends AbstractDataSet[T, RDD[T]] {

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DataSet[C] = {
    val preDataSet = this

    val broadcast = this.originRDD().sparkContext.broadcast(transformer)

    val cachedTransformer =
      preDataSet.originRDD().mapPartitions(_ => Iterator
        .single(broadcast.value.cloneTransformer())
      ).setName("Cached Transformer").persist()

    new DistributedDataSet[C] {
      override def size(): Long = preDataSet.size()

      override def shuffle(): Unit = preDataSet.shuffle()

      override def data(train: Boolean): RDD[C] =
        preDataSet.data(train).zipPartitions(cachedTransformer)(
          (data, tran) => tran.next()(data))

      override def originRDD(): RDD[_] = preDataSet.originRDD()

      override def cache(): Unit = {
        cachedTransformer.count()
        isCached = true
      }

      override def unpersist(): Unit = {
        cachedTransformer.unpersist()
        isCached = false
      }
    }
  }

  /**
   * Get the 'origin' RDD of the dataset.
   *
   * @return
   */
  def originRDD(): RDD[_]

  /**
   * Trigger the computation of this dataset and cache it in memory.
   */
  def cache(): Unit = {
    if (originRDD() != null) {
      originRDD().count()
    }
    isCached = true
  }

  /**
   * Unpersist rdd.
   */
  def unpersist(): Unit = {
    if (originRDD() != null) {
      originRDD().unpersist()
      isCached = false
    }
  }

  /**
   * Check if rdd is cached.
   */
  var isCached = false

  override def getSplits(): Seq[DistributedDataSet[T]] = Seq(this)
}

/**
 * Wrap DataSet as a splittable dataset with inherited transformer which can transform itself into another DataSet.
 * @param preDataSet the dataset before splitting
 * @param transformer the transformer used for transform
 * @param ev$1
 * @param ev$2
 * @tparam S dataType of the former DataSet
 * @tparam C dataType of the transformed DataSet after calling transform()
 */
class SplittableTransformedDistributedDataSet[S: ClassTag, C: ClassTag]
(preDataSet: DistributedDataSet[S], transformer: Transformer[S, C])
  extends DistributedDataSet[C] with Serializable {

  val broadcast: Broadcast[Transformer[S, C]] =
    preDataSet.originRDD().sparkContext.broadcast(transformer)

  val cachedTransformer: RDD[Transformer[S, C]] = preDataSet.originRDD()
    .mapPartitions(_ => Iterator.single(broadcast.value.cloneTransformer())
    ).setName("Cached Transformer").persist()

  override def size(): Long = preDataSet.size()

  override def shuffle(): Unit = preDataSet.shuffle()

  override def originRDD(): RDD[_] = preDataSet.originRDD()

  override def cache(): Unit = {
    cachedTransformer.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    cachedTransformer.unpersist()
    preDataSet.unpersist()
    isCached = false
  }

  override def data(train: Boolean): RDD[C] = {
    DataSet.logger.debug(s"-------called data($train) in $this ------------")
    preDataSet.data(train).zipPartitions(cachedTransformer)(
      (data, tran) => tran.next()(data))
  }

  override def getSplits(): Seq[DistributedDataSet[C]] = {
      val preDataSets = preDataSet.getSplits()
      preDataSets.map(pre => new SplittableTransformedDistributedDataSet(pre, transformer.cloneTransformer()))
  }

  override def transform[M: ClassTag](transformer: Transformer[C, M]): DataSet[M] = {
    new SplittableTransformedDistributedDataSet[C, M](this, transformer)
  }
}

/**
 * Wrap a RDD as a DataSet.
 * @param buffer
 * @param isInOrder whether need keeping original data order, default false
 * @param groupSize offset range, from 0 until buffer.length - groupSize + 1,
 *                  only use when need keep original order
 * @tparam T
 */
class CachedDistriDataSet[T: ClassTag] private[dataset]
(buffer: RDD[Array[T]], isInOrder: Boolean = false, groupSize: Int = 1)
  extends DistributedDataSet[T] with Serializable {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single((0 until iter.next().length).toArray)
  }).setName("original index").cache()

  override def data(train: Boolean): RDD[T] = {
    DataSet.logger.debug(s"-------called data($train) in $this ------------")
    val _train = train
    val _groupSize = if (isInOrder) Utils.getBatchSize(groupSize) else 1
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexOffset = math.max(1, indexes.length - (_groupSize - 1))
      val localData = dataIter.next()
      val offset = if (_train) {
        RandomGenerator.RNG.uniform(0, indexOffset).toInt
      } else {
        0
      }
      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    if (!isInOrder) {
      indexes.unpersist()
      indexes = buffer.mapPartitions(iter => {
        Iterator.single(RandomGenerator.shuffle((0 until iter.next().length).toArray))
      }).setName("shuffled index").cache()
    }
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }

  override def ->[C: ClassTag](transformer: Transformer[T, C]): DistributedDataSet[C] = {
    this.transform(transformer)
  }

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DistributedDataSet[C] = {
    new SplittableTransformedDistributedDataSet(this, transformer)
  }

}

/**
 * Common used DataSet builder.
 */
object DataSet {
  val logger = Logger.getLogger(getClass)

  /**
    * Indicate whether to coalesce the original input data.
    */
  var declineRepartitionedRdd = true

  /**
    * Decide how to deal with user defined repartitioned RDDs.
    * @return
    */
  def declineRepartition(accept: Boolean): Unit = {
    declineRepartitionedRdd = accept
  }

  /**
   * Wrap an array as a DataSet.
   */
  def array[T](data: Array[T]): LocalArrayDataSet[T] = {
    new LocalArrayDataSet[T](data)
  }

  /**
   * Wrap an array as a distributed DataSet.
   * @param localData
   * @param sc
   * @tparam T
   * @return
   */
  def array[T: ClassTag](localData: Array[T], sc: SparkContext): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      sc.parallelize(localData, nodeNumber)
        // Keep this line, or the array will be send to worker every time
        .coalesce(nodeNumber, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    )
  }

  /**
   * Wrap a RDD as a DataSet.
   * @param data
   * @tparam T
   * @return
   */
  def rdd[T: ClassTag](data: RDD[T]): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    val transformedData = if (declineRepartitionedRdd) {
      new CachedDistriDataSet[T](
        data.coalesce(nodeNumber, true)
          .mapPartitions(iter => {
            Iterator.single(iter.toArray)
          }).setName("cached dataset")
          .cache()
      )
    } else {
      val originalRdd = data

      // should not call methods other than transform() and getSplits(), for that will be meaningless
      new DistributedDataSet[T] with Serializable {
        override def originRDD(): RDD[_] = originalRdd
        override def data(train: Boolean): RDD[T] = ???
        override def shuffle(): Unit = ???
        override def size(): Long = ???

        override def transform[C: ClassTag](transformer: Transformer[T, C]): DataSet[C] = {
          new SplittableTransformedDistributedDataSet(this, transformer)
        }

        override def getSplits(): Seq[CachedDistriDataSet[T]] = {
          if (declineRepartitionedRdd) {
            Seq(new CachedDistriDataSet[T](
              originalRdd.coalesce(nodeNumber, true)
                .mapPartitions(iter => {
                  Iterator.single(iter.toArray)
                }).setName("cached dataset")
                .cache()
            ))
          } else {
            val nodeNum = Engine.nodeNumber()
            val partitions = originalRdd.partitions.length
            val splits = partitions / nodeNum + (if (partitions % nodeNum == 0) 0 else 1)

            val rddBatches = mutable.ArrayBuffer[CachedDistriDataSet[T]]()
            for (i <- 0 until splits) {
              val splittedData = originalRdd
                .mapPartitionsWithIndex((index, iter) => {
                  val upperLimit = (i + 1) * nodeNum
                  val downLimit = upperLimit - nodeNum
                  if (index < upperLimit && index >= downLimit) iter
                  else Iterator.empty
                })
              rddBatches.append(new CachedDistriDataSet(
                splittedData.repartition(nodeNum)
                  .mapPartitions(iter => {
                  Iterator.single(iter.toArray)
                }).setName("cached dataset")
                  .cache()
              ))
            }
            rddBatches
          }
        }
      }
    }
    transformedData
  }

  def imageFrame(imageFrame: ImageFrame): DataSet[ImageFeature] = {
    imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        rdd[ImageFeature](distributedImageFrame.rdd)
      case localImageFrame: LocalImageFrame =>
        array(localImageFrame.array)
    }
  }

  /**
   * Wrap a RDD as a DataSet.
   * @param data
   * @tparam T
   * @return
   */
  private[bigdl] def sortRDD[T: ClassTag](data: RDD[T], isInOrder: Boolean = false,
                                          groupSize: Int = 1): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    val transformedData = if (declineRepartitionedRdd) data.coalesce(nodeNumber, true) else data
    new CachedDistriDataSet[T](
      transformedData
        .mapPartitions(iter => {
          Iterator.single(sortData(iter.toArray, isInOrder))
        }).setName("cached dataset")
        .cache(),
      isInOrder,
      groupSize
    )
  }

  /**
   * sort data from small to big, only support Sample data type.
   * @param data original data
   * @param isInOrder whether to sort data by ascending order
   * @return
   */
  def sortData[T: ClassTag](data: Array[T], isInOrder: Boolean): Array[T] = {
    if (isInOrder) {
      require(classTag[T] == classTag[Sample[_]],
        "DataSet.sortData: Only support sort for sample input")
      data.sortBy(a => a.asInstanceOf[Sample[_]].featureLength(0))
    } else {
      data
    }
  }

  /**
   * Generate a DataSet from a local image folder. The image folder should have two levels. The
   * first level is class folders, and the second level is images. All images belong to a same
   * class should be put into the same class folder. So each image in the path is labeled by the
   * folder it belongs.
   */
  object ImageFolder {
    /**
     * Extract all image paths into a Local DataSet. The paths are all labeled. When the image
     * files are too large(e.g. ImageNet2012 data set), you'd better readd all paths instead of
     * image files themselves.
     * @param path
     * @return
     */
    def paths(path: Path): LocalDataSet[LocalLabeledImagePath] = {
      val buffer = LocalImageFiles.readPaths(path)
      new LocalArrayDataSet[LocalLabeledImagePath](buffer)
    }

    /**
     * Extract all images under the given path into a Local DataSet. The images are all labeled.
     * @param path
     * @param scaleTo
     * @return
     */
    def images(path: Path, scaleTo: Int): DataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val total = paths.length
      var count = 1
      val buffer = paths.map(imageFile => {
        if (total < 100 || count % (total / 100) == 0 || count == total) {
          logger.info(s"Cache image $count/$total(${count * 100 / total}%)")
        }
        count += 1

        val bufferBGR = new LabeledBGRImage()
        bufferBGR.copy(BGRImage.readImage(imageFile.path, scaleTo), 255f)
          .setLabel(imageFile.label)
      })

      new LocalArrayDataSet[LabeledBGRImage](buffer)
    }

    /**
     * Extract all images under the given path into a Distributed DataSet. The images are all
     * labeled.
     * @param path
     * @param sc
     * @param scaleTo
     * @return
     */
    def images(path: Path, sc: SparkContext, scaleTo: Int)
    : DataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val buffer: Array[LabeledBGRImage] = {
        paths.map(imageFile => {
          val bufferBGR = new LabeledBGRImage()
          bufferBGR.copy(BGRImage.readImage(imageFile.path, scaleTo), 255f)
            .setLabel(imageFile.label)
        })
      }
      array(buffer, sc)
    }
  }

  /**
   * Create a DataSet from a Hadoop sequence file folder.
   */
  object SeqFileFolder {
    val logger = Logger.getLogger(getClass)

    /**
     * Extract all hadoop sequence file paths from a local file folder.
     * @param path
     * @param totalSize
     * @return
     */
    def paths(path: Path, totalSize: Long): LocalDataSet[LocalSeqFilePath] = {
      logger.info(s"Read sequence files folder $path")
      val buffer: Array[LocalSeqFilePath] = findFiles(path)
      logger.info(s"Find ${buffer.length} sequence files")
      require(buffer.length > 0, s"Can't find any sequence files under $path")
      new LocalArrayDataSet[LocalSeqFilePath](buffer) {
        override def size(): Long = {
          totalSize
        }
      }
    }

    /**
     * get label from text of sequence file,
     * @param data text of sequence file, this text can split into parts by "\n"
     * @return
     */
    def readLabel(data: Text): String = {
      val dataArr = data.toString.split("\n")
      if (dataArr.length == 1) {
        dataArr(0)
      } else {
        dataArr(1)
      }
    }

    /**
     * get name from text of sequence file,
     * @param data text of sequence file, this text can split into parts by "\n"
     * @return
     */
    def readName(data: Text): String = {
      val dataArr = data.toString.split("\n")
      require(dataArr.length >= 2, "key in seq file only contains label, no name")
      dataArr(0)
    }

    /**
     * Extract hadoop sequence files from an HDFS path
     * @param url
     * @param sc
     * @param classNum
     * @return
     */
    def files(url: String, sc: SparkContext, classNum: Int): DistributedDataSet[ByteRecord] = {
      val nodeNumber = Engine.nodeNumber()
      val coreNumber = Engine.coreNumber()
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text],
        nodeNumber * coreNumber).map(image => {
        ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
      }).filter(_.label <= classNum)

      rdd[ByteRecord](rawData)
    }

    /**
     * Extract hadoop sequence files from an HDFS path as RDD
     * @param url sequence files folder path
     * @param sc spark context
     * @param classNum class number of data
     * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
     * @return
     */
    private[bigdl] def filesToRdd(url: String, sc: SparkContext,
      classNum: Int, partitionNum: Option[Int] = None): RDD[ByteRecord] = {
      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text], num).map(image => {
        ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
      }).filter(_.label <= classNum)
      val transformedData = if (declineRepartitionedRdd) rawData.coalesce(num, true) else rawData
//      rawData.coalesce(num, true)
      transformedData
    }

    /**
     * Extract hadoop sequence files from an HDFS path as ImageFrame
     * @param url sequence files folder path
     * @param sc spark context
     * @param classNum class number of data
     * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
     * @return
     */
    private[bigdl] def filesToImageFrame(url: String, sc: SparkContext,
      classNum: Int, partitionNum: Option[Int] = None): ImageFrame = {
      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text], num).map(image => {
        val rawBytes = image._2.copyBytes()
        val label = Tensor[Float](T(readLabel(image._1).toFloat))
        val imgBuffer = ByteBuffer.wrap(rawBytes)
        val width = imgBuffer.getInt
        val height = imgBuffer.getInt
        val bytes = new Array[Byte](3 * width * height)
        System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
        val imf = ImageFeature(bytes, label)
        imf(ImageFeature.originalSize) = (height, width, 3)
        imf
      }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
      ImageFrame.rdd(rawData)
    }

    private[bigdl] def filesToImageFeatureDataset(url: String, sc: SparkContext,
      classNum: Int, partitionNum: Option[Int] = None): DistributedDataSet[ImageFeature] = {
      rdd[ImageFeature](filesToImageFrame(url, sc, classNum, partitionNum).toDistributed().rdd)
    }

    private[bigdl] def findFiles(path: Path): Array[LocalSeqFilePath] = {
      val directoryStream = Files.newDirectoryStream(path)
      import scala.collection.JavaConverters._
      directoryStream.asScala.map(_.toAbsolutePath.toString)
        .filter(_.endsWith(".seq")).toArray.sortWith(_ < _).map(p => LocalSeqFilePath(Paths.get(p)))
    }

  }

}




