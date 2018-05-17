## NNImageReader

`NNImageReader` is the primary DataFrame-based image loading interface, defining API to read images
into DataFrame.

Scala:
```scala
    val imageDF = NNImageReader.readImages(imageDirectory, sc)
```

Python:
```python
    image_frame = NNImageReader.readImages(image_path, self.sc)
```

The output DataFrame contains a sinlge column named "image". The schema of "image" column can be
accessed from `com.intel.analytics.zoo.pipeline.nnframes.DLImageSchema.byteSchema`.
Each record in "image" column represents one image record, in the format of
Row(origin, height, width, num of channels, mode, data), where origin contains the URI for the image file,
and `data` holds the original file bytes for the image file. `mode` represents the OpenCV-compatible
type: CV_8UC3, CV_8UC1 in most cases.
```scala
  val byteSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_8UC3, CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // Bytes in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", BinaryType, false) :: Nil)
```

After loading the image, user can compose the preprocess process with the `Preprocessing` defined
in `com.intel.analytics.zoo.feature.image`.
