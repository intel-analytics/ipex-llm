## Trigger ##

**Scala:**
```scala
Trigger.severalIteration(iterationNum)
```
**Python:**
```python
MaxEpoch(endTriggerNum)
```

or

```
MaxIteration(endTriggerNum)
```

A trigger specifies a timespot or several timespots during training,
and a corresponding action will be taken when the timespot(s)
s reached.

1. everyEpoch

   A trigger that triggers an action when each epoch finishs.
   Could be used as trigger in setValidation and setCheckpoint
   in Optimizer, and also in TrainSummary.setSummaryTrigger.
   
2. severalIteration

    A trigger that triggers an action every "n" iterations.
    Could be used as trigger in setValidation and setCheckpoint
    in Optimizer, and also in TrainSummary.setSummaryTrigger.
    
3. maxEpoch

   A trigger that triggers an action when training reaches
   the number of epochs specified by "max".
   Usually used in Optimizer.setEndWhen.

4. maxIteration

    A trigger that triggers an action when training reaches
    the number of iterations specified by "max".
    Usually used in Optimizer.setEndWhen.


