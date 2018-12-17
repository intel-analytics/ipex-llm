It's recommended to use protobuffer based serialization method `saveModule` when you want to persist your trained
models or models loaded from third party frameworks, details could be found from [Module API](../../APIGuide/Module.md).
This article illustrates the implementations and how to implement customized serializer for your layers

### BigDL serialziation hierarchy
Below is the class hierarchy for the serialization framework

                         Loadable       Savable
                             .              .
                             .              .
                             ................
                                     .
                                     .
                            ModuleSerializable
                                     .
                                     .
            ........................................................................
            .               .              .                       .               . 
            .               .              .                       .               .
     ModuleSerializer CellSerializer  ContainerSerializable   KerasSerializer  [UserDefinedSerializer]
                                           .
                                           .
                               ..........................
                               .                        .
                               .                        .
                         ContainerSerializer  [UserDefinedContainerSerializer] 
                                        
                                  
                             
        
 
 * ModuleSerializable: abstract class to define serialization methods and provide a default implementation
 * ModuleSerializer: entry for all layers' serializers
 * CellSerializer: Default implementation for RNN cell modules
 * ContainerSerializable: Abstract class for container serializers like Sequential and provide a default implementation
 * ContainerSerializer: Default serializer for containers
 * KerasSerializer: Keras adapter serializer implementation for keras compatible layers
 
 Users can extend ModuleSerializable or ContainerSerializable to implement optional serializers for your own layer or containers
 
### Supported data types

Below are the data types supported in serialization

* Int
* Long
* Short
* Float
* Double
* Boolean
* String
* Regularizer
* DataFormat
* VariableFormat
* Shape
* InitializationMethod
* Tensor
* Module
* Array
* Map
* Customized Data

#### Implement customized data converter
if you have your own defined data types that are not supported in serialization
or cannot be indirectly supported by above types, you can also define your own data
converter by extending trait `DataConverter`, which has two abstract methods to implement

The `setAttributeValue` is to define how to set your own object value to attributeBuilder

```scala
 def setAttributeValue[T : ClassTag](context: SerializeContext[T],
                                      attributeBuilder : AttrValue.Builder, value: Any,
                                      valueType: universe.Type = null)
    (implicit ev: TensorNumeric[T]) : Unit

```

In opposite you should implement `getAttributeValue` to get value from attibute

```scala
 def getAttributeValue[T : ClassTag](context: DeserializeContext,
                                      attribute: AttrValue)(
    implicit ev: TensorNumeric[T]) : AnyRef
```
Check [BigDL](https://github.com/intel-analytics/BigDL)`com.intel.analytics.bigdl.utils.serializer.converters.DataConverter`
to see more details

Then register your data converter in `DataConverter`

```scala
def registerConverter(tpe : String, converter : DataConverter) : Unit 
```

`tpe` is the `scala.reflect.Type` string representation

### Implement customized serializer

As described above, BigDL provides a default serializer which works for most layers, thus we don't need to write serializer
for these layers. But there are some layers which are not stateless (Note : `stateless` here means except for parameters like weight and bias, and fields from layer constructor, there are no other fields that their values could change and the layer will behavior differently with these values)
 
 To implement a customized serializer is straightforward, you just need to define a new serializer by extending trait `ModuleSerializable` For most cases, you just need to override two methods
 
 `doSerializeModule` defines how you serialize the stateful variables (besides weights and bias), if you layer has construct fields types of which are supported 
 by BigDL, you don't even need to explicitly manage then, you could just call `super.doSerializeModule(context, bigDLModelBuilder)` instead for these values.
 
 ```scala
 protected def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                               bigDLModelBuilder : BigDLModule.Builder)
                                              (implicit ev: TensorNumeric[T]) : Unit
```

`doLoadModule` defines how you deserialize the statefule variables, same as serialization, if you layer has construct fields types of which are supported 
by BigDL, you don't even need to explicitly manage then, you could just call `super.doSerializeModule(context)` instead 

```scala
protected def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T]
```

The only thing you want to enable your serializer is to register it in `ModuleSerializer`

```scala
def registerModule(moduleType : String, serializer : ModuleSerializable) : Unit 
```

`ModuleType` is the full classpath of your layer and the serializer is the serializer object you just defined

Similarly,  if you want to define a new serializer for your containers, you just need to define your own serializer by 
extending `ContainerSerializable` and override the same two methods above