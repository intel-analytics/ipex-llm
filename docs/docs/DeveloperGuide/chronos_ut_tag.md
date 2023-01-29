# Add tags to Chronos unit-test

This page demonstrates how to add different tags to Chronos unit-tests.

## Motivation

Chronos provides different installation options ([pytorch], [tensorflow], [automl], [inference], [distributed], [all]) but our unit tests are mixed together. In fact, this causes some trouble for our nightly tests and installation tests.

Now, we have added six ops to divide uts according to different installation options:

·`op_torch`: corresponds to bigdl-chronos[pytorch]

·`op_tf2`: corresponds to bigdl-chronos[tensorflow]

·`op_automl`: corresponds to bigdl-chronos[automl]

·`op_distributed`: corresponds to bigdl-chronos[distributed]

·`op_inference`: corresponds to bigdl-chronos[inference]

·`op_diff_set_all`: corresponds to dependencies after removing bigdl-chronos[pytorch,tensorflow,automl,inference,distributed] in bigdl-chronos[all] (i.e. pmdarima, prophet, tsfresh and pyarrow)

## How to add tags

When we add a new ut, corresponding tags should be added.

**Basic principle**:
The ut can be passed only after installing the required options, then corresponding tags are necessary.

Specifically, some uts may not specify backend (pytorch or tensorflow), such as `test_deduplicate_timeseries_dataframe` in "/test/bigdl/chronos/data/utils/test_deduplicate.py", then `op_torch` and `op_tf2` both are needed.

## Example

For example, `test_fit_np` in "/test/bigdl/chronos/autots/model/test_auto_lstm.py" can be passed only after installing `bigdl-chronos[pytorch,distributed]`, then `op_torch` and `op_distributed` two tags are necessary.

```python
@op_distributed
class TestAutoLSTM(TestCase):
    def setUp(self) -> None:
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    @op_torch
    def test_fit_np(self):
```

This ut is deselected when run `pytest -v -m "torch and not distributed" /test/bigdl/chronos/autots/`. Meanwhile, this ut is selected when run `pytest -v -m "torch and distributed" /test/bigdl/chronos/autots/` or `pytest -v -m "torch" /test/bigdl/chronos/autots/`.

> **Note**:
> 
> If all tests of one class require same tag, we can directly add the tag to the class instead of adding to all functions, just like `@op_distributed` in above example.