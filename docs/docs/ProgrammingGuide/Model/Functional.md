BigDL supports two different model definition styles: Sequential API and Functional API.

In Functional API, the model is described as a graph. It is more convenient than Sequential API
when define some complex model.

---
## **Define a simple model**
Suppose we want to define a model with three layers
```
Linear -> Sigmoid -> Softmax
```

You can write code like this

**Scala:**
```scala
val linear = Linear(...).inputs()
val sigmoid = Sigmoid().inputs(linear)
val softmax = Softmax().inputs(sigmoid)
val model = Graph(Seq[linear], Seq[softmax])
```
**Python:**
```python
linear = Linear(...)()
sigmoid = Sigmoid()(linear)
softmax = Softmax()(sigmoid)
model = Model([linear], [softmax])
```

An easy way to understand the Funtional API is to think of each layer in the model as a directed
edge connecting its input and output

In the above code, first we create an input node named as linear by using
the Linear layer, then connect it to the sigmoid node with a Sigmoid
layer, then connect the sigmoid node to the softmax node with a Softmax layer.

After defined the graph, we create the model by passing in the input nodes
and output nodes.

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

**Scala:**
```scala
val linear1 = Linear(...).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(...).inputs(relu1)
val relu2 = ReLU().inputs(linear2)
val linear3 = Linear(...).inputs(relu1)
val relu3 = ReLU().inputs(linear3)
val model = Graph(Seq[linear1], Seq[relu2, relu3])
```
**Python:**
```python
linear1 = Linear(...)()
relu1 = ReLU()(linear1)
linear2 = Linear(...)(relu1)
relu2 = ReLU()(linear2)
linear3 = Linear(...)(relu1)
relu3 = ReLU()(linear3)
model = Model(Seq[linear1], Seq[relu2, relu3])
```
In the above node, linear2 and linear3 are both from relu1 with separated
Linear layers, which construct the branch structure. When we create the model,
the outputs parameter contains relu2 and relu3 as the model has two outputs.

---
## **Define a model with merged branch**
Suppose we want to define a model like this
```
Linear -> ReLU --> Linear -> ReLU ----> Add
               |-> Linear -> ReLU --|
```
In the model, the outputs of the two branches are merged by an add operation.

You can define the model like this

**Scala:**
```scala
val linear1 = Linear(...).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(...).inputs(relu1)
val relu2 = ReLU().inputs(linear2)
val linear3 = Linear(...).inputs(relu1)
val relu3 = ReLU().inputs(linear3)
val add = CAddTable().inputs(relu2, relu3)
val model = Graph(Seq[linear1], Seq[add])
```
**Python:**
```python
linear1 = Linear(...)()
relu1 = ReLU()(linear1)
linear2 = Linear(...)(relu1)
relu2 = ReLU()(linear2)
linear3 = Linear(...)(relu1)
relu3 = ReLU()(linear3)
add = CAddTable()(relu2, relu3)
model = Model(Seq[linear1], Seq[add])
```
In the above code, to merge the branch, we use the CAddTable, which takes two
input nodes, to generate one output node.

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

You can define the model like this

**Scala:**
```scala
val linear1 = Linear(...).inputs()
val relu1 = ReLU().inputs(linear1)
val linear2 = Linear(...).inputs()
val relu2 = ReLU().inputs(linear2)
val add = CAddTable().inputs(relu1, relu2)
val model = Graph(Seq[linear1, linear2], Seq[add])
```
**Python:**
```python
linear1 = Linear(...)()
relu1 = ReLU()(linear1)
linear2 = Linear(...)()
relu2 = ReLU()(linear2)
add = CAddTable()(relu1, relu2)
model = Model(Seq[linear1, linear2], Seq[add])
```
In the above code, we define two input nodes linear1 and linear2 and put them
into the first parameter when create the graph model.

