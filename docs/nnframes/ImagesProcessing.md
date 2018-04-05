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

## NNImageTransformer

`NNImageTransformer` provides DataFrame-based API for image pre-processing and feature transformation.
`NNImageTransformer` follows the Spark Transformer API pattern and can be used as one stage in
Spark ML pipeline.

The input column can be either DLImageSchema.byteSchema or DLImageSchema.floatSchema. If
using NNImageReader, the default format is DLImageSchema.byteSchema The output column is always
 DLImageSchema.floatSchema.

`NNImageTransformer` takes BigDL transfomer as the contructor params.
Scala:
```
val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
  ChannelNormalize(123, 117, 104, 1, 1, 1)
val transformedDF = new NNImageTransformer(transformer)
  .setInputCol("image")
  .setOutputCol("features")
  .transform(imageDF)
```
Python:
```python
image_frame = NNImageReader.readImages(self.image_path, self.sc)
transformer = NNImageTransformer(
    Pipeline([Resize(256, 256), CenterCrop(224, 224),
              ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
              MatToTensor()])
).setInputCol("image").setOutputCol("output")

result = transformer.transform(image_frame)
```