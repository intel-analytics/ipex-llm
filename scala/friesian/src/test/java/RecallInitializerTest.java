import com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.recall.IndexService;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class RecallInitializerTest {

    @Test
    void testInitialization() throws IOException, InterruptedException {
        String configPath = "/home/xingyuan/projects/serving/BigDL/scala/friesian/src/test/resources/nearlineConfig/config_recall.yaml";
        RecallInitializer.main(new String[]{"-c", configPath});
        // you can get initialDataPath file from friesian-serving.tar.gz

        SparkSession sparkSession = SparkSession.builder().getOrCreate();
        Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().getInitialDataPath());

        IndexService indexService = new IndexService(NearlineUtils.helper().getIndexDim());
        indexService.load(NearlineUtils.helper().getIndexPath());

        assertEquals(dataset.count(), indexService.getNTotal());
        Map<Integer, double[]> vectorMap = new HashMap<>();
        List<Row> rows = dataset.collectAsList();
        rows.forEach(row -> vectorMap.put(row.getInt(0), ((DenseVector) row.get(1)).toArray()));
        rows.forEach(row -> {
            double[] doubleArr = ((DenseVector) row.get(1)).toArray();
            float[] floatArr = new float[doubleArr.length];
            for (int i = 0; i < doubleArr.length; i++) {
                floatArr[i] = (float) doubleArr[i];
            }
            assertArrayEquals(vectorMap.get(row.getInt(0)),
                    vectorMap.get(indexService.search(IndexService.vectorToFloatArray(floatArr), 1)[0]));

//            assertEquals(indexService.search(IndexService.vectorToFloatArray(floatArr), 1)[0], row.getInt(0));
//            When multi vector have same value, you may get unexpected id
        });
    }
}
