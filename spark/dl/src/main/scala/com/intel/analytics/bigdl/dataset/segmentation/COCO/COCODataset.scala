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

package com.intel.analytics.bigdl.dataset.segmentation.COCO

import com.google.gson.{Gson, GsonBuilder, JsonDeserializationContext, JsonDeserializer, JsonElement, TypeAdapter}
import com.google.gson.annotations.SerializedName
import com.google.gson.reflect.TypeToken
import com.google.gson.stream.{JsonReader, JsonWriter}
import java.io.{BufferedReader, FileReader}
import java.lang.reflect.Type
import java.nio.ByteBuffer
import scala.collection.mutable.ArrayBuffer

private[COCO] class COCOSerializeContext {
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


private[COCO] class COCODeserializer(buffer: ByteBuffer) {
  private def getFloat: Float = buffer.getFloat
  private def getDouble: Double = buffer.getDouble
  private def getInt: Int = buffer.getInt
  private def getLong: Long = buffer.getLong
  private def getBoolean: Boolean = buffer.get != 0

  def getString: String = {
    val len = getInt
    val arr = new Array[Byte](len)
    buffer.get(arr)
    new String(arr)
  }
  case class SimpleAnnotation(categoryId: Int, area: Float, bbox1: Float, bbox2: Float,
    bbox3: Float, bbox4: Float, isCrowd: Boolean, rleCounts: Array[Float])

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
    val rle = if (isCrowd) {
      // is RLE
      val countLen = getInt
      val arr = new Array[Float](countLen)
      for (i <- 0 until countLen) {
        arr(i) = MaskAPI.uint2long(getInt)
      }
      arr
    } else {
      val firstDimLen = getInt
      val poly = new Array[Array[Double]](firstDimLen)
      for (i <- 0 until firstDimLen) {
        val secondDimLen = getInt
        val inner = new Array[Double](secondDimLen)
        for (j <- 0 until secondDimLen) {
          inner(j) = getDouble
        }
        poly(i) = inner
      }
      val cocoRLE = MaskAPI.mergeRLEs(MaskAPI.poly2RLE(COCOPoly(poly), height, width), false)
      cocoRLE.counts.map(MaskAPI.uint2long(_).toFloat)
    }
    SimpleAnnotation(categoryId, area, bbox1, bbox2, bbox3, bbox4, isCrowd, rle)
  }
}

case class COCODataset(info: COCOInfo, images: Array[COCOImage],
  annotations: Array[COCOAnotationOD],
  licenses: Array[COCOLicence], categories: Array[COCOCategory]) {

  private lazy val cateId2catIdx = scala.collection.mutable.Map[Long, Int]()
  def init(): Unit = {
    val id2img = images.toIterator.map(img => (img.id, img)).toMap
    annotations.foreach(anno => {
      require(id2img.contains(anno.imageId), s"Cannot find image_id ${anno.imageId}")
      val img = id2img(anno.imageId)
      anno.image = img
      img.annotations += anno
    })
    categories.zipWithIndex.foreach { case (cate, idx) =>
      cateId2catIdx(cate.id) = idx
    }
  }

  def categoryId2Idx(id: Long): Int = cateId2catIdx(id)
}

case class COCOInfo(
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

}

case class COCOAnotationOD(id: Long, imageId: Long, categoryId: Long,
  segmentation: COCOSegmentation, area: Float, bbox: (Float, Float, Float, Float), isCrowd: Boolean,
  @transient var image: COCOImage = null
) {
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

abstract class COCOSegmentation{
  def dumpTo(context: COCOSerializeContext): Unit
}

case class COCOPoly(poly: Array[Array[Double]]) extends COCOSegmentation {
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

 case class COCORLE(counts: Array[Int], height: Int, width: Int) extends COCOSegmentation {
   /**
    * Get an element in the counts. Process the overflowed int
    *
    * @param idx
    * @return
    */
   def get(idx: Int): Long = {
     MaskAPI.uint2long(counts(idx))
   }

   override def dumpTo(context: COCOSerializeContext): Unit = {
     context.dump(counts.length)
     counts.foreach(p => {
       context.dump(p)
     })
   }
 }

object COCODataset {

  private[COCO] class AnnotationDeserializer extends
    JsonDeserializer[COCOAnotationOD] {
    private lazy val intArrAdapter = COCODataset.gson.getAdapter(classOf[Array[Int]])
    private lazy val polyAdapter = COCODataset.gson.getAdapter(classOf[Array[Array[Double]]])
    override def deserialize(json: JsonElement, ty: Type,
      context: JsonDeserializationContext): COCOAnotationOD = {
      val obj = json.getAsJsonObject
      val id = obj.get("id").getAsLong
      val imageId = obj.get("image_id").getAsLong
      val categoryId = obj.get("category_id").getAsLong
      val area = obj.get("area").getAsFloat
      val rawBbox = obj.get("bbox").getAsJsonArray
      require(rawBbox.size() == 4, "The bbox in the COCO annotation data should have 4 elements")
      val bbox = (rawBbox.get(0).getAsFloat, rawBbox.get(1).getAsFloat, rawBbox.get(2).getAsFloat,
        rawBbox.get(3).getAsFloat)
      val isCrowd = if (obj.get("iscrowd").getAsInt == 1) true else false
      val seg = if (isCrowd) {
        val segJson = obj.getAsJsonObject("segmentation")
        val cnts = intArrAdapter.fromJsonTree(segJson.get("counts"))
        val size = intArrAdapter.fromJsonTree(segJson.get("size"))
        require(size.length == 2, "The size in the COCO annotation data should have 2 elements")
        COCORLE(cnts, size(0), size(1))
      } else {
        val polys = polyAdapter.fromJsonTree(obj.get("segmentation"))
        COCOPoly(polys)
      }
      COCOAnotationOD(id, imageId, categoryId, seg, area, bbox, isCrowd)
    }
  }

  lazy val gson = {
    val gsonBuilder = new GsonBuilder()
    val theType = new TypeToken[COCOAnotationOD]() {}.getType
    val deserializer = new AnnotationDeserializer
    gsonBuilder.registerTypeAdapter(theType, deserializer)
    gsonBuilder.create()
  }

  def load(path: String): COCODataset = {
    val d = gson.fromJson(
      new BufferedReader(new FileReader(path)), classOf[COCODataset])
    d.init()
    d
  }

  def main(args: Array[String]): Unit = {
    val ds = load("/home/menooker/work/coco/instances_val2014.json")


    println(ds.licenses(0).name)
  }
}
