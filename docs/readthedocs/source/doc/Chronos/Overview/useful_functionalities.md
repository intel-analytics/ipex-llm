# Distributed Processing


#### Distributed training
LSTM, TCN and Seq2seq users can easily train their forecasters in a distributed fashion to **handle extra large dataset and utilize a cluster**. The functionality is powered by Project Orca.
```python
f = Forecaster(..., distributed=True)
f.fit(...)
f.predict(...)
f.to_local()  # collect the forecaster to single node
f.predict_with_onnx(...)  # onnxruntime only supports single node
```
#### Distributed Data processing: XShardsTSDataset
```eval_rst
.. warning::
    ``XShardsTSDataset`` is still experimental.
```
`TSDataset` is a single thread lib with reasonable speed on large datasets(~10G). When you handle an extra large dataset or limited memory on a single node, `XShardsTSDataset` can be involved to handle the exact same functionality and usage as `TSDataset` in a distributed fashion.

```python
# a fully distributed forecaster pipeline
from orca.data.pandas import read_csv
from bigdl.chronos.data.experimental import XShardsTSDataset

shards = read_csv("hdfs://...")
tsdata, _, test_tsdata = XShardsTSDataset.from_xshards(...)
tsdata_xshards = tsdata.roll(...).to_xshards()
test_tsdata_xshards = test_tsdata.roll(...).to_xshards()

f = Forecaster(..., distributed=True)
f.fit(tsdata_xshards, ...)
f.predict(test_tsdata_xshards, ...)
```
