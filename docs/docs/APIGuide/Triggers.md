---
A trigger specifies a timespot or several timespots during training,
and a corresponding action will be taken when the timespot(s)
s reached.

---
## Every Epoch
**Scala:**
```scala
 val trigger = Trigger.everyEpoch
```
**Python:**
```python
 trigger = EveryEpoch()
```
   A trigger that triggers an action when each epoch finishes.
   Could be used as trigger in `setValidation` and `setCheckpoint`
   in Optimizer, and also in `TrainSummary.setSummaryTrigger`.

---
## Several Iteration

**Scala:**
```scala
 val trigger = Trigger.severalIteration(n)
```
**Python:**

```python
 trigger = SeveralIteration(n)
```

 A trigger that triggers an action every `n` iterations.
 Could be used as trigger in `setValidation` and `setCheckpoint` 
 in Optimizer, and also in `TrainSummary.setSummaryTrigger`.

---    
## Max Epoch
**Scala:**
```scala
 val trigger = Trigger.maxEpoch(max)
```
**Python:**
```python
 trigger = MaxEpoch(max)
``` 

  A trigger that triggers an action when training reaches
  the number of epochs specified by "max".
  Usually used in `Optimizer.setEndWhen`.

---
## Max Iteration

**Scala:**
```scala
 val trigger = Trigger.maxIteration(max)
```
**Python:**
```python
 trigger = MaxIteration(max)
``` 

  A trigger that triggers an action when training reaches
  the number of iterations specified by "max".
  Usually used in `Optimizer.setEndWhen`.

---    
## Max Score
**Scala:**
```scala
 val trigger = Trigger.maxScore(max)
```
**Python:**
```python
 trigger = MaxScore(max)
``` 

  
 A trigger that triggers an action when validation score
 larger than "max" score

---
## Min Loss
**Scala:**
```scala
 val trigger = Trigger.minLoss(min)
```
**Python:**
```python
 trigger = MinLoss(min)
``` 

  
 A trigger that triggers an action when training loss
 less than "min" loss

---
## And
**Scala:**
```scala
  val trigger = Trigger.and(Trigger.minLoss(0.01), Trigger.maxEpoch(2))
```
**Python:**
```python
  trigger = TriggerAnd(MinLoss(0.01), MaxEpoch(2))
``` 
A trigger contains other triggers and triggers when all of them trigger (logical AND). 

For example, TriggerAnd(MinLoss(0.01), MaxEpoch(2)), means training should stop when loss < 0.01 and epoch > 2.

---
## Or
**Scala:**
```scala
  val trigger = Trigger.and(Trigger.minLoss(0.01), Trigger.maxEpoch(2))
```
**Python:**
```python
  trigger = TriggerOr(MinLoss(0.01), MaxEpoch(2))
``` 
A trigger contains other triggers and triggers when all of them trigger (logical OR). 

For example, TriggerOr(MinLoss(0.01), MaxEpoch(2)), means training should stop when loss < 0.01 or epoch > 2.
