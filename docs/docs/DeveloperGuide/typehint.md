# Python Type hint

This page describes how to add type annotations to unmarked code efficiently.

## Introduction

We need to declare that the python runtime does not enforce function and variable [type annotations](https://docs.python.org/3/library/typing.html#module-typing). But they can actually be used by third party tools such as type checkers, IDEs, linters, etc.

**Python Enhancement Proposals(PEPs)** are widely accepted python standards, and [PEP 484](https://peps.python.org/pep-0484/) introduced syntax for function annotations, for example:
```python
def greeting(name: str) -> str:
    return 'Hello ' + name
```
This states that the expected type of the name argument is str. Analogically, the expected return type is str.

Expressions whose type is a subtype of a specific argument type are also accepted for that argument.

## MonkeyType: Automatic Annotation

MonkeyType is a python tools collects runtime types of function arguments and return values, and can automatically generate stub files or even add draft type annotations directly to your Python code based on the types collected at runtime. More details at [MonkeyType github](https://github.com/Instagram/MonkeyType#example).

We can collect the types of interface exposed to the user by running unit tests with monkeytype, but we recommend achieving this by previewing the changes first with `monkey stub` and then manually applying them.

## How Do We Work on Type Hint
Takes bigdl.friesian.feature as an example.

0. Preparation
```shell
pip install monkeytype

cd Bigdl/python
source friesian/dev/prepare_env.sh
```

1. Collect runtime types with monkeytype
Since monkeytype can only operate file-by-file, we shall run uts individually or Or automate the process with the help of `add_type_hint.sh`.
```shell
#  Usage:
#   bash dev/add_type_hint.sh module_name submodule_name
#   module_names: orca, dllib, chronos, friesian, etc.
#   submodule_name: directories under module_names/test/bigdl/module_names
#  
#  Example:
#   `bash dev/add_type_hint.sh friesian feature` will run all unit tests 
#   under python/friesian/test/bigdl/friesian/feature

bash dev/add_type_hint.sh friesian feature
```
If all UTs pass, you will see `friesian_hint.sqlite3` under `dev/`, which contains stubs of refactors. Check the list of all modules which have traces present in the trace store:
```shell
> export MT_DB_PATH="dev/friesian_hint.sqlite3" # monkeytype will read this traces database
> monkeytype list-modules
test.bigdl.friesian.feature.test_table
test.bigdl.friesian.feature.conftest
pyspark.traceback_utils
... ...
bigdl.friesian.feature.utils
bigdl.friesian.feature.table
... ...
```

2. Run `monkeytype stub some.module` to generate a stub file for the given module based on call traces queried from the trace store.
```shell
> monkeytype stub bigdl.friesian.feature.table

...
class Table:
    def __init__(self, df: "SparkDataFrame") -> None: ...

    @property
    def schema(self) -> "StructType": ...

    @staticmethod
    def _read_parquet(paths: Union[List[str], str]) -> "SparkDataFrame": ...
...

```
which indicates the annotations involved in the UTs. More usages about stub see [docs](https://monkeytype.readthedocs.io/en/latest/generation.html#monkeytype-stub).

3. Apply type annotaions manually (recommended) or use `monkeytype apply some.module`.

4. **Manually** check again if annotations **are** consistent

## Notes
1. `monkeytype apply` may not work for some cases. 

For example, `friesian.feature.table` invokes two kinds of DataFrame in this module:` pyspark.sql.dataframe.DataFrame `and `pandas.core.frame.DataFrame`. To avoid ambiguity of type `DataFrame`, we rename `pyspark.sql.dataframe.DataFrame` to SparkDataFrame like:
```
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
```
And so do PandasDataFrame.

2. Use [TYPE_CHECKING](https://docs.python.org/3/library/typing.html#constant) constant to avoid import unnecessary libraries at runtime.
```python
if TYPE_CHECKING:
    from pandas.core.frame import DataFrame as PandasDataFrame
    from pyspark.sql.column import Column
    from pyspark.sql import Row
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType
```
3. Mark `TODO` if there are methods not caught by UTs.
