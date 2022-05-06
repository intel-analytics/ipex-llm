BigDL provides error handling api. Please don't use `assert`, `raise`, `throw` to fail the application.
Use the error handling api instead, it will provide useful message for debugging.

## **Error handling API**

**Scala**

Import package
```scala
import com.intel.analytics.bigdl.dllib.utils.Log4Error
```

If user's input is invalid
```scala
Log4Error.invalidInputError(condition: Boolean, errmsg: String, fixmsg: String = null)
```

* `condition`: Will throw exception and print errmsg if it's `false`. Otherwise will pass the function
* `errmsg`: Error message to be print.
* `fixmsg`: Message about how to fix the error.

If user call the api wrong
```scala
Log4Error.invalidOperationError(condition: Boolean, errmsg: String, fixmsg: String = null, cause: Throwable = null)
```

* `condition`: Will throw exception and print errmsg if it's `false`. Otherwise will pass the function
* `errmsg`: Error message to be print.
* `fixmsg`: Message about how to fix the error.
* `cause`: Exception need to throw.

For unkown Exception:
```scala
Log4Error.unKnowExceptionError(condition: Boolean, errmsg: String, fixmsg: String = null, cause: Throwable = null)
```

* `condition`: Will throw exception and print errmsg if it's `false`. Otherwise will pass the function
* `errmsg`: Error message to be print.
* `fixmsg`: Message about how to fix the error.
* `cause`: Exception need to throw.

Notes: This API is for future extension, in case we need distinct invalidOperation exception with unKnownException in python


**Python**

Import package
```python
from bigdl.dllib.utils.log4Error import *
```

If user's input is invalid
```python
invalidInputError(condition, errMsg, fixMsg=None)
```

* `condition`: Will throw exception and print errmsg if it's `false`. Otherwise will pass the function
* `errMsg`: Error message to be print.
* `fixMsg`: Message about how to fix the error.

If user call the api wrong
```python
invalidOperationError(condition, errMsg, fixMsg=None, cause=None)
```

* `condition`: Will throw exception and print errmsg if it's `false`. Otherwise will pass the function
* `errMsg`: Error message to be print.
* `fixMsg`: Message about how to fix the error.
* `cause`: Exception need to throw.

---
## **Examples**
**Scala**

If you want to use:
```scala
assert(a > 0)
```
or
```scala
require(a > 0)
```
or
```scala
if (a <= 0) {
  throw new Exception()
}
```
Please use below code instead
```scala
Log4Error.invalidInputError(a>0, errmsg="a is negative", fixmsg="expect a is positive")
```

**Python**

If you want to use:
```python
assert(a > 0)
```
or
```python
if (a <= 0) {
  raise Exception()
}
```
Please use below code instead
```python
invalidInputError(a>0, errMsg="a is negative", fixMsg="expect a is positive")
```
