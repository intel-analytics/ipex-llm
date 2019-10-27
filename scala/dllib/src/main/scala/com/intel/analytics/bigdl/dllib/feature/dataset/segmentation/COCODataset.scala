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
import java.io.{BufferedReader, FileReader}
import java.lang.reflect.Type
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}
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
  case class SimpleAnnotation(categoryId: Int, area: Float, bbox1: Float, bbox2: Float,
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
    val categoryId = getInt
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
    SimpleAnnotation(categoryId, area, bbox1, bbox2, bbox3, bbox4, isCrowd, masks)
  }
}

case class COCODataset(info: COCODatasetInfo, images: Array[COCOImage],
  annotations: Array[COCOAnotationOD],
  licenses: Array[COCOLicence], categories: Array[COCOCategory]) {

  private lazy val cateId2catIdx = scala.collection.mutable.Map[Long, Int]()
  private lazy val imageId2Image = images.toIterator.map(img => (img.id, img)).toMap

  private[segmentation] def init(imgRoot: String): Unit = {
    annotations.foreach(anno => {
      require(imageId2Image.contains(anno.imageId), s"Cannot find image_id ${anno.imageId}")
      val img = imageId2Image(anno.imageId)
      anno.image = img
      img.annotations += anno
      anno.segmentation match {
        case poly: COCOPoly =>
          anno.segmentation = COCOPoly(poly.poly, img.height, img.width)
        case _ =>
      }
    })
    images.foreach(_.imgRootPath = imgRoot)
    categories.zipWithIndex.foreach { case (cate, idx) =>
      cateId2catIdx(cate.id) = idx + 1 // the ids starts from 1, because 0 is for background
    }
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

  def dumpTo(context: COCOSerializeContext, dataset: COCODataset): Unit = {
    context.dump(height)
    context.dump(width)
    context.dump(annotations.size)
    annotations.foreach(_.dumpTo(context, dataset))
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

  def dumpTo(context: COCOSerializeContext, dataSet: COCODataset): Unit = {
    context.dump(dataSet.categoryId2Idx(categoryId))
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
   * Load COCO dataset
   * @param jsonPath the JSON metadata file path
   * @param imageRoot the root path of the image files
   * @return
   */
  def load(jsonPath: String, imageRoot: String = "."): COCODataset = {
    val d = gson.fromJson(
      new BufferedReader(new FileReader(jsonPath)), classOf[COCODataset])
    d.init(imageRoot)
    d
  }
}
