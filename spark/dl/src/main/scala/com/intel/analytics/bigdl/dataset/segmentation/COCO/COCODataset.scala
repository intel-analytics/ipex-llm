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
import scala.collection.mutable.ArrayBuffer

case class COCODataset(info: COCOInfo, images: Array[COCOImage],
  annotations: Array[COCOAnotationOD],
  licenses: Array[COCOLicence], categories: Array[COCOCategory]) {
  def init(): Unit = {
    val id2img = images.toIterator.map(img => (img.id, img)).toMap
    annotations.foreach(anno => {
      require(id2img.contains(anno.imageId), s"Cannot find image_id ${anno.imageId}")
      val img = id2img(anno.imageId)
      anno.image = img
      img.annotations += anno
    })
  }
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
}

case class COCOAnotationOD(id: Long, imageId: Long, categoryId: Long,
  segmentation: COCOSegmentation, area: Float, bbox: (Float, Float, Float, Float), isCrowd: Boolean,
  @transient var image: COCOImage = null
)

case class COCOLicence(
  id: Long, name: String, url: String
)

case class COCOCategory(
  id: Long, name: String) {
  @SerializedName("supercategory") var superCategory: String = _
}

class COCOSegmentation{}

case class COCOPoly(poly: Array[Array[Double]]) extends COCOSegmentation

case class COCORLE(counts: Array[Int], height: Int, width: Int) extends COCOSegmentation {
  /**
   * Get an element in the counts. Process the overflowed int
   * @param idx
   * @return
   */
  def get(idx: Int): Long = {
    MaskAPI.uint2long(counts(idx))
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
