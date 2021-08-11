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

package com.intel.analytics.bigdl.dataset.segmentation

import com.google.gson.{Gson, GsonBuilder, JsonDeserializationContext, JsonDeserializer, JsonElement, TypeAdapter}
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.{JsonReader, JsonWriter}
import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, RoiImageInfo}
import com.intel.analytics.bigdl.utils.{T, Table}
import java.io.{BufferedReader, FileReader}
import java.lang.reflect.Type
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer

private[bigdl] class COCOSerializeContext {
  private val converter4 = ByteBuffer.allocate(4)
  private val converter8 = ByteBuffer.allocate(8)
  private val buffer = new ArrayBuffer[Byte]()

  def dump(v: Float): Unit = {
    converter4.clear()
    converter4.putFloat(v)
    buffer.appendAll(converter4.array())
  }

  def dump(v: Double): Unit = {
    converter8.clear()
    converter8.putDouble(v)
    buffer.appendAll(converter8.array())
  }


  def dump(v: Int): Unit = {
    converter4.clear()
    converter4.putInt(v)
    buffer.appendAll(converter4.array())
  }

  def dump(v: Long): Unit = {
    converter8.clear()
    converter8.putLong(v)
    buffer.appendAll(converter8.array())
  }

  def dump(v: Boolean): Unit = {
    val d: Byte = if (v) 1 else 0
    buffer.append(d)
  }

  def dump(v: String): Unit = {
    val bytes = v.getBytes
    dump(bytes)
  }

  def clear(): Unit = buffer.clear()

  def dump(v: Array[Byte]): Unit = {
    dump(v.length)
    buffer.appendAll(v)
  }

  def toByteArray: Array[Byte] = buffer.toArray
}


private[bigdl] class COCODeserializer(buffer: ByteBuffer) {
  private def getFloat: Float = buffer.getFloat
  private def getDouble: Double = buffer.getDouble
  def getInt: Int = buffer.getInt
  private def getLong: Long = buffer.getLong
  private def getBoolean: Boolean = buffer.get != 0

  def getString: String = {
    val len = getInt
    val arr = new Array[Byte](len)
    buffer.get(arr)
    new String(arr)
  }
  case class SimpleAnnotation(categoryIdx: Int, area: Float, bbox1: Float, bbox2: Float,
    bbox3: Float, bbox4: Float, isCrowd: Boolean, masks: SegmentationMasks)

  // returns an image's height, width, all annotations
  def getAnnotations: (Int, Int, Array[SimpleAnnotation]) = {
    val height = getInt
    val width = getInt
    val nAnnotations = getInt
    val anno = (0 until nAnnotations).map(_ => getAnnotation(height, width))
    (height, width, anno.toArray)
  }

  private def getAnnotation(height: Int, width: Int): SimpleAnnotation = {
    val categoryIdx = getInt
    val area = getFloat
    val bbox1 = getFloat
    val bbox2 = getFloat
    val bbox3 = getFloat
    val bbox4 = getFloat
    val isCrowd = getBoolean
    val masks = if (isCrowd) {
      // is RLE
      val countLen = getInt
      val arr = new Array[Int](countLen)
      for (i <- 0 until countLen) {
        arr(i) = getInt
      }
      RLEMasks(arr, height, width)
    } else {
      val firstDimLen = getInt
      val poly = new Array[Array[Float]](firstDimLen)
      for (i <- 0 until firstDimLen) {
        val secondDimLen = getInt
        val inner = new Array[Float](secondDimLen)
        for (j <- 0 until secondDimLen) {
          inner(j) = getFloat
        }
        poly(i) = inner
      }
      PolyMasks(poly, height, width)
    }
    SimpleAnnotation(categoryIdx, area, bbox1, bbox2, bbox3, bbox4, isCrowd, masks)
  }
}

case class COCODataset(info: COCODatasetInfo, images: Array[COCOImage],
  annotations: Array[COCOAnotationOD],
  licenses: Array[COCOLicence], categories: Array[COCOCategory]) {

  private lazy val cateId2catIdx = scala.collection.mutable.Map[Long, Int]()
  private lazy val imageId2Image = images.toIterator.map(img => (img.id, img)).toMap

  private[segmentation] def init(imgRoot: String): Unit = {
    categories.zipWithIndex.foreach { case (cate, idx) =>
      cateId2catIdx(cate.id) = idx + 1 // the ids starts from 1, because 0 is for background
    }
    annotations.foreach(anno => {
      require(imageId2Image.contains(anno.imageId), s"Cannot find image_id ${anno.imageId}")
      val img = imageId2Image(anno.imageId)
      anno._categoryIdx = cateId2catIdx(anno.categoryId)
      anno.image = img
      img.annotations += anno
      anno.segmentation match {
        case poly: COCOPoly =>
          anno.segmentation = COCOPoly(poly.poly, img.height, img.width)
        case _ =>
      }
    })
    images.foreach(_.imgRootPath = imgRoot)

  }

  /**
   * Find a COCOImage by the image id
   * @param id image id
   * @return the COCOImage with the given id
   */
  def getImageById(id: Long): COCOImage = imageId2Image(id)

  /**
   * Convert COCO categoryId into category index.
   * COCO dataset's categoryId is not continuous from 1 to number of categories.
   * This function maps every categoryId to a number from 1 to number of categories - The result is
   * called category index. The category index 0 is reserved for "background" class.
   * @param id categoryId
   * @return category index
   */
  def categoryId2Idx(id: Long): Int = cateId2catIdx(id)

  /**
   * Get the category data by the category index
   * @param idx category index
   * @return category data
   */
  def getCategoryByIdx(idx: Int): COCOCategory = categories(idx - 1)

  /**
   * Convert the images & ground truths into ImageFeatures.
   * The image feature is in the same format of what COCODataset.loadFromSeqFile returns
   * @return
   */
  def toImageFeatures: Iterator[ImageFeature] = images.toIterator.map(_.toImageFeature)
}

case class COCODatasetInfo(
  year: Int,
  version: String,
  description: String,
  contributor: String,
  url: String
  ) {
  @SerializedName("date_created") var dateCreated: String = _
}

case class COCOImage(
  id: Long,
  height: Int,
  width : Int,
  license: Int
) {
  @transient lazy val annotations: ArrayBuffer[COCOAnotationOD] = new ArrayBuffer[COCOAnotationOD]
  @transient private[segmentation] var imgRootPath: String = _
  @SerializedName("flickr_url") var flickrUrl: String = _
  @SerializedName("coco_url") var cocoUrl: String = _
  @SerializedName("date_captured") var dateCaptured: String = _
  @SerializedName("file_name") var fileName: String = _

  def dumpTo(context: COCOSerializeContext): Unit = {
    context.dump(height)
    context.dump(width)
    context.dump(annotations.size)
    annotations.foreach(_.dumpTo(context))
  }

  /**
   * Get the path of the image in local file system
   * @return
   */
  def path: Path = Paths.get(imgRootPath, fileName)

  /**
   * Read the data from the image file
   * @return
   */
  def data: Array[Byte] = Files.readAllBytes(path)

  /**
   * Convert the image's image data and ground truth into an image feature.
   * The image feature is in the same format of what COCODataset.loadFromSeqFile returns
   * @return an ImageFeature with ground truth & image data
   */
  def toImageFeature: ImageFeature = {
    val labelClasses = Tensor(annotations.map(_.categoryIdx.toFloat).toArray,
      Array(annotations.length))
    val bboxes = Tensor(
      annotations.toIterator.flatMap(ann => {
        val x1 = ann.bbox._1
        val y1 = ann.bbox._2
        val x2 = ann.bbox._3
        val y2 = ann.bbox._4
        Iterator(x1, y1, x2, y2)
      }).toArray,
      Array(annotations.length, 4))
    val isCrowd = Tensor(annotations.map(ann => if (ann.isCrowd) 1f else 0f).toArray,
      Array(annotations.length))
    val masks = annotations.map(ann => ann.segmentation.asInstanceOf[SegmentationMasks]).toArray

    val rawdata = SeqFileFolder.decodeRawImageToBGR(this.data)
    require(rawdata.length == height * width * 3)
    val imf = ImageFeature(rawdata, RoiLabel(labelClasses, bboxes, masks), fileName)
    imf(ImageFeature.originalSize) = (height, width, 3)
    imf(RoiImageInfo.ISCROWD) = isCrowd
    imf
  }

  /**
   * Convert the image's ground truth label & masks into Table for RoiMiniBatch
   * @return a table with ground truth label & masks for the image
   */
  def toTable: Table = {
    val img = this
    val bboxes = Tensor(
      img.annotations.toIterator.flatMap(ann => {
        val x1 = ann.bbox._1
        val y1 = ann.bbox._2
        val x2 = ann.bbox._3
        val y2 = ann.bbox._4
        Iterator(x1, y1, x2, y2)
      }).toArray,
      Array(img.annotations.length, 4))

    T()
      .update(RoiImageInfo.ISCROWD,
        Tensor(img.annotations.map(ann => if (ann.isCrowd) 1f else 0f).toArray,
          Array(img.annotations.length))
      )
      .update(RoiImageInfo.ORIGSIZE, (img.height, img.width, 3))
      .update(RoiImageInfo.MASKS,
        img.annotations.map(ann => ann.segmentation.asInstanceOf[SegmentationMasks].toRLE).toArray)
      .update(RoiImageInfo.BBOXES, bboxes)
      .update(RoiImageInfo.CLASSES,
        Tensor(img.annotations.map(ann => ann.categoryIdx.toFloat).toArray,
          Array(img.annotations.length)))
  }
}

/**
 * An annotation for an image (OD in the name for Object Detection)
 * @param id
 * @param imageId the Id of the image
 * @param categoryId the Id of the category. Note that categoryId is not continuous from 0 to
 *                   the number of categories. You can use COCODataset.cateId2Idx to convert an
 *                   categoryId to a compact category index.
 * @param segmentation the segmentation data
 * @param area  area
 * @param bbox  the bounding box, (xmin, ymin, xmax, ymax)
 * @param isCrowd if the annotation is a crowd. e.g. a crowd of people. If true, segmentation is
 *                an COCORLE object
 * @param image the reference to the image
 */
case class COCOAnotationOD(id: Long, imageId: Long, categoryId: Long,
  var segmentation: COCOSegmentation, area: Float,
  bbox: (Float, Float, Float, Float), isCrowd: Boolean, @transient var image: COCOImage = null) {

  @transient private[segmentation] var _categoryIdx: Long = -1
  def categoryIdx: Long = _categoryIdx

  def dumpTo(context: COCOSerializeContext): Unit = {
    require(_categoryIdx != -1, "COCOAnotationOD should be initialized")
    context.dump(_categoryIdx.toInt)
    context.dump(area)
    context.dump(bbox._1)
    context.dump(bbox._2)
    context.dump(bbox._3)
    context.dump(bbox._4)
    context.dump(isCrowd)
    segmentation.dumpTo(context)
  }
}

case class COCOLicence(
  id: Long, name: String, url: String
)

case class COCOCategory(
  id: Long, name: String) {
  @SerializedName("supercategory") var superCategory: String = _
}

trait COCOSegmentation {
  def dumpTo(context: COCOSerializeContext): Unit
}

case class COCOPoly(_poly: Array[Array[Float]], _height: Int, _width: Int)
  extends PolyMasks(_poly, _height, _width) with COCOSegmentation {
  override def dumpTo(context: COCOSerializeContext): Unit = {
    context.dump(poly.length)
    poly.foreach(p => {
      context.dump(p.length)
      p.foreach(xy => {
        context.dump(xy)
      })
    })
  }
}

 case class COCORLE(_counts: Array[Int], _height: Int, _width: Int)
   extends RLEMasks(_counts, _height, _width) with COCOSegmentation {
   override def dumpTo(context: COCOSerializeContext): Unit = {
     context.dump(counts.length)
     counts.foreach(p => {
       context.dump(p)
     })
   }
 }

object COCODataset {
  private[bigdl] val MAGIC_NUM = 0x1f3d4e5a
  private[segmentation] class AnnotationDeserializer extends
    JsonDeserializer[COCOAnotationOD] {
    private lazy val intArrAdapter = COCODataset.gson.getAdapter(classOf[Array[Int]])
    private lazy val polyAdapter = COCODataset.gson.getAdapter(classOf[Array[Array[Float]]])
    override def deserialize(json: JsonElement, ty: Type,
      context: JsonDeserializationContext): COCOAnotationOD = {
      val obj = json.getAsJsonObject
      val id = obj.get("id").getAsLong
      val imageId = obj.get("image_id").getAsLong
      val categoryId = obj.get("category_id").getAsLong
      val area = obj.get("area").getAsFloat
      val rawBbox = obj.get("bbox").getAsJsonArray
      require(rawBbox.size() == 4, "The bbox in the COCO annotation data should have 4 elements")
      val (x1, y1, w, h) = (rawBbox.get(0).getAsFloat, rawBbox.get(1).getAsFloat,
        rawBbox.get(2).getAsFloat, rawBbox.get(3).getAsFloat)
      val bbox = (x1, y1, x1 + w - 1, y1 + h - 1)
      val isCrowd = if (obj.get("iscrowd").getAsInt == 1) true else false
      val seg = if (isCrowd) {
        val segJson = obj.getAsJsonObject("segmentation")
        val cnts = intArrAdapter.fromJsonTree(segJson.get("counts"))
        val size = intArrAdapter.fromJsonTree(segJson.get("size"))
        require(size.length == 2, "The size in the COCO annotation data should have 2 elements")
        COCORLE(cnts, size(0), size(1))
      } else {
        val polys = polyAdapter.fromJsonTree(obj.get("segmentation"))
        COCOPoly(polys, -1, -1)
      }
      COCOAnotationOD(id, imageId, categoryId, seg, area, bbox, isCrowd)
    }
  }

  private lazy val gson = {
    val gsonBuilder = new GsonBuilder()
    val theType = new TypeToken[COCOAnotationOD]() {}.getType
    val deserializer = new AnnotationDeserializer
    gsonBuilder.registerTypeAdapter(theType, deserializer)
    gsonBuilder.create()
  }

  /**
   * Load COCO dataset from local file
   * @param jsonPath the JSON metadata file path
   * @param imageRoot the root path of the image files
   * @return the loaded COCO dataset
   */
  def load(jsonPath: String, imageRoot: String = "."): COCODataset = {
    val d = gson.fromJson(
      new BufferedReader(new FileReader(jsonPath)), classOf[COCODataset])
    d.init(imageRoot)
    d
  }

  /**
   * Load COCO dataset from Hadoop sequence files
   * @param url sequence files folder path on HDFS/Local
   * @param sc spark context
   * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
   * @return ImageFeatures for the dataset.
   *         Key in ImageFeature    Value               Type
   *         ImageFeature.bytes     decoded image data  Array[Byte]
   *         ImageFeature.uri       Image file name     String
   *         ImageFeature.label     Label & masks       RoiLabel
   *         RoiImageInfo.ISCROWD   isCrowd             Tensor[Float]
   */
  def loadFromSeqFile(url: String, sc: SparkContext,
    partitionNum: Option[Int] = None): DataSet[ImageFeature] = {
    SeqFileFolder.filesToRoiImageFeatures(url, sc, partitionNum)
  }
}
