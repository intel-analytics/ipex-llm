## **Transformer**

 Transformer is for pre-processing. In many deep learning workload, input data need to be pre-processed before fed into   model. For example, in CNN, the image file need to be decoded from some compressed format(e.g. jpeg) to float arrays,    normalized and cropped to some fixed shape. You can also find pre-processing in other types of deep learning work        load(e.g. NLP, speech recognition). In BigDL, we provide many pre-process procedures for user. They're implemented as    Transformer.

 The transformer interface is
```scala

trait Transformer[A, B] extends Serializable {
   def apply(prev: Iterator[A]): Iterator[B]
 }
```

 It's simple, right? What a transformer do is convert a sequence of objects of Class A to a sequence of objects of Class  B.

 Transformer is flexible. You can chain them together to do pre-processing. Let's still use the CNN example, say first    we need read image files from given paths, then extract the image binaries to array of float, then normalized the image  content and crop a fixed size from the image at a random position. Here we need 4 transformers, `PathToImage`,           `ImageToArray`, `Normalizor` and `Cropper`. And then chain them together.
