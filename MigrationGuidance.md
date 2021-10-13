## Migration Guidance

This guidance is used to provide guidance to BigDL and Analytics zoo users to migrate their existing BigDL/Analytics Zoo applications to use BigDL2.0

* **For BigDL users**

   ***scala application***

   Change ```import com.intel.analytics.bigdl.XYZ``` to ```import com.intel.analytics.bigdl.dllib.XYZ```

    except the following:

   ```com.intel.analytics.bigdl.dataset.XYZ``` to ```com.intel.analytics.bigdl.dllib.feature.dataset.XYZ```

   ```com.intel.analytics.bigdl.transform.XYZ``` to ```com.intel.analytics.bigdl.dllib.feature.transform.XYZ```
   
   ```com.intel.analytics.bigdl.nn.keras.XYZ``` is deprecated and will be removed. Pleaase use zoo keras api instead

   If you are a maven user and add BigDL as dependency to your own project. Please change the dependency as :
   ```
   <dependency>
       <groupId>com.intel.analytics.bigdl</groupId>
       <artifactId>bigdl-dllib_{spark_version}</artifactId>
       <version>${BIGDL_VERSION}</version>
   </dependency>
   ```

   If you are a sbt user, please change libraryDependencies to:
   ```
   libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-dllib" % "${BIGDL_VERSION}"
   ```

   ***python application***

    Change ```from bigdl.XYZ import *``` to ```from bigdl.dllib.XYZ import *```

    except the following:

   ```from bigdl.dataset.XYZ import *``` to ```from bigdl.dllib.feature.dataset.XYZ import *```

   ```from bigdl.transform.XYZ import *``` to ```from bigdl.dllib.feature.transform.XYZ import *```

   ```bigdl.nn.keras.XYZ``` is deprecated and will be removed. Please use zoo keras api instead

* **For Analytics Zoo users**

   ***scala application***

   ****feature/common/nnframes modules****

   Change ```import com.intel.analytics.zoo.XYZ``` to ```import com.intel.analytics.bigdl.dllib.XYZ```

   ****Keras modules****

   Change ```import com.intel.analytics.zoo.pipeline.api.keras.XYZ``` to ```import com.intel.analytics.bigdl.dllib.keras.XYZ```
   
   ****Estimator modules****

   Change ```import com.intel.analytics.zoo.pipeline.estimator.XYZ``` to ```import com.intel.analytics.bigdl.dllib.estimator.XYZ```

If you are a maven user and add above modules as dependency to your own project. Please change the dependency as
   ```
   <dependency>
       <groupId>com.intel.analytics.bigdl</groupId>
       <artifactId>bigdl-dllib_{spark_version}</artifactId>
       <version>${BIGDL_VERSION}</version>
   </dependency>
   ```

   If you are a sbt user, please change libraryDependencies to:
   ```
   libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-dllib" % "${BIGDL_VERSION}"
   ```

   ****tfpark modules****

   Change ```import com.intel.analytics.zoo.XYZ``` to ```import com.intel.analytics.bigdl.orca.XY$

   If you are a maven user and add above modules as dependency to your own project. Please change the dependency as :
   ```
   <dependency>
       <groupId>com.intel.analytics.bigdl</groupId>
       <artifactId>bigdl-orca_{spark_version}</artifactId>
       <version>${BIGDL_VERSION}</version>
   </dependency>
   ```

   If you are a sbt user, please change libraryDependencies to:
   ```
   libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-orca" % "${BIGDL_VERSION}"
   ```

   ***python application***

   ****feature/common/nnframes modules****

   Change ```from zoo.XYZ import *``` to ```from bigdl.dllib.XYZ import *```

   except the following:

   ```from zoo.util.XYZ import *``` to ```from bigdl.dllib.utils.XYZ import *```

   ```from zoo.common.XYZ import *``` to ```from bigdl.dllib.utils.XYZ import *```

   ****Keras modules****

   Change ```from zoo.pipeline.api.keras import *``` to ```from bigdl.dllib.keras import *```

   ****Estimator modules****

   Change ```from zoo.pipeline.api.estimator import *``` to ```from bigdl.dllib.estimator import *```

   ****tfpark modules****

  Change ```from zoo.XYZ import *``` to ```from bigdl.orca.XYZ import *```

