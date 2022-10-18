package com.intel.analytics.bigdl.friesian;

import com.intel.analytics.bigdl.friesian.serving.feature.utils.LettuceUtils;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.lang.reflect.Field;
import java.util.Arrays;

public class JavaTestUtils {
    public static void destroyLettuceUtilsInstance() throws NoSuchFieldException, IllegalAccessException {
        Field field = LettuceUtils.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);
    }

    public static Seq<String> convertListToSeq(String[] inputArray) {
        return JavaConverters.asScalaIteratorConverter(Arrays.asList(inputArray).iterator()).asScala().toSeq();
    }
}
