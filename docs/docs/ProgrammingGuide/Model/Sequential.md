BigDL supports two different model definition styles: Sequential API and Functional API.

Here we introduce how to define a model in Sequential API.

---
## **Define a simple model**
Suppose we want to define a model with three layers
```
Linear -> Sigmoid -> SoftMax
```

You can write code like this

**Scala:**
```scala
import com.intel.analytics.bigdl.nn._

val model = Sequential()
model.add(Linear(3, 5))
model.add(Sigmoid())
model.add(SoftMax())
```
**Python:**
```python
model = Sequential()
model.add(Linear(3, 5))
model.add(Sigmoid())
model.add(SoftMax())
```

In the above code, we first create a container Sequential. Then add the layers
into the container one by one. The order of the layers in the model is same with the insertion
order. This model definition
looks very straightforward.

BigDL provides multiple types of contianers allow user to define complex model in sequential
style. We will take a look at it.

---
## **Define a model with branches**
Suppose we want to define a model like this
```
Linear -> ReLU --> Linear -> ReLU
               |-> Linear -> ReLU
```
The model has two outputs from two branches. The inputs of the branches are both the
output from the first ReLU.

You can define the model like this

**Scala**
```scala
val branch1 = Sequential().add(Linear(...)).add(ReLU())
val branch2 = Sequential().add(Linear(...)).add(ReLU())
val branches = ConcatTable().add(branch1).add(branch2)

val model = Sequential()
model.add(Linear(...))
model.add(ReLU())
model.add(branches)
```

**Python**
```python
branch1 = Sequential().add(Linear(...)).add(ReLU())
branch2 = Sequential().add(Linear(...)).add(ReLU())
branches = ConcatTable().add(branch1).add(branch2)

val model = Sequential()
model.add(Linear(...))
model.add(ReLU())
model.add(branches)
```
In the above code, to handle the branch structure, we use another container ConcatTable.
When you add layers into ConcatTable, the new layer won't be placed after the previous one
but will become a new branch.

The input of the model is a tensor and the output of the model is two tensors.

---
## **Define a model with merged branch**
Suppose we want to define a model like this
```
Linear -> ReLU --> Linear -> ReLU ----> Add
               |-> Linear -> ReLU --|
```
In the model, the outputs of the two branches are merged by an add operation.

You can define the model like this

**Scala**
```scala
val branch1 = Sequential().add(Linear(...)).add(ReLU())
val branch2 = Sequential().add(Linear(...)).add(ReLU())
val branches = ConcatTable().add(branch1).add(branch2)

val model = Sequential()
model.add(Linear(...))
model.add(ReLU())
model.add(branches)
model.add(CAddTable())
```

**Python**
```python
branch1 = Sequential().add(Linear(...)).add(ReLU())
branch2 = Sequential().add(Linear(...)).add(ReLU())
branches = ConcatTable().add(branch1).add(branch2)

val model = Sequential()
model.add(Linear(...))
model.add(ReLU())
model.add(branches)
model.add(CAddTable())
```
To merge the outputs of the branches by an add operation, we use CAddTable. It
takes a list of tensors from the previous layer, and merge the tensors by adding them together.

BigDL provides many merge layers. Please check Merge layers document page. They all
take a list of tensors as input and merge the tensors by some operation.

---
## **Define a model with multiple inputs**
We have already seen how to define branches in model and how to merge branches.
What if we have multiple input? Suppose we want to define a model like this
```
Linear -> ReLU ----> Add
Linear -> ReLU --|
```

The above model takes two tensors as input, and merge them together by add operation.

You can define the model like this

**Scala**
```scala
val model = Sequential()
val branches = ParallelTable()
val branch1 = Sequential().add(Linear(...)).add(ReLU())
val branch2 = Sequential().add(Linear(...)).add(ReLU())
branches.add(branch1).add(branch2)
model.add(branches).add(CAddTable())
```

**Python**
```python
model = Sequential()
branches = ParallelTable()
branch1 = Sequential().add(Linear(...)).add(ReLU())
branch2 = Sequential().add(Linear(...)).add(ReLU())
branches.add(branch1).add(branch2)
model.add(branches).add(CAddTable())
```

In the above code, we use ParallelTable to handle the multiple inputs. ParallelTable also
define a multiple branches structure like ConcatTable. The difference is it takes a list
of tensors as inputs and assign each tensor to the corresponding branch.

