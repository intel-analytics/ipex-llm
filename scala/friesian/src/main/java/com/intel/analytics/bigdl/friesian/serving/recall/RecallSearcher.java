package com.intel.analytics.bigdl.friesian.serving.recall;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.floatArray;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.feature.FeatureUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.recall.RecallUtils;
import io.grpc.stub.StreamObserver;

public class RecallSearcher {
    private IndexService indexService;
    private FeatureGrpc.FeatureBlockingStub featureServiceStub;
    private Timer overallTimer;
    private Timer predictTimer;
    private Timer faissTimer;

    public RecallSearcher(FeatureGrpc.FeatureBlockingStub featureServiceStub,
                          Timer overallTimer,
                          Timer predictTimer,
                          Timer faissTimer){

        indexService = new IndexService(Utils.helper().indexDim());
        assert(Utils.helper().getIndexPath() != null): "indexPath must be provided";
        indexService.load(Utils.helper().getIndexPath());
        System.out.printf("Index service nTotal = %d\n", this.indexService.getNTotal());
        this.featureServiceStub = featureServiceStub;
        this.overallTimer = overallTimer;
        this.predictTimer = predictTimer;
        this.faissTimer = faissTimer;

     }

    public RecallProto.Candidates SearchByItemId(RecallProto.ItemQuery itemQuery) throws Exception {
        Timer.Context overallContext = overallTimer.time();
        int itemID = itemQuery.getItemID();
        int k = itemQuery.getK();

        Timer.Context predictContext = predictTimer.time();
        float[] itemFeatureList;
        FeatureProto.IDs itemIds = FeatureProto.IDs.newBuilder().addID(itemID).build();
        FeatureProto.Features feature = featureServiceStub.getItemFeatures(itemIds);
        Object[][] featureList = FeatureUtils.getFeatures(feature);
        if (featureList[0] == null) {
            throw new Exception("Can't get user feature from feature service");
        }
        itemFeatureList = RecallUtils.featureObjToFloatArr(featureList[0]);
        predictContext.stop();
        Timer.Context faissContext = faissTimer.time();
        int[] candidates =
                indexService.search(IndexService.vectorToFloatArray(itemFeatureList), k);
        faissContext.stop();
        RecallProto.Candidates.Builder result = RecallProto.Candidates.newBuilder();
        // TODO: length < k
        for (int i = 0; i < k; i ++) {
            result.addCandidate(candidates[i]);
        }
        overallContext.stop();
        return result.build();

    }


    public RecallProto.Candidates SearchByCategory(RecallProto.CategoryQuery query) {
        Timer.Context overallContext = overallTimer.time();
        int catID = query.getCatID();
        String name = query.getCatName();
        int k = query.getK();
        Timer.Context predictContext = predictTimer.time();
        return null;
        // search from redis
    }


    public RecallProto.Candidates SearchByEmbbed(RecallProto.Query query) throws Exception {
        return search(query);
    }

    public RecallProto.Candidates search(RecallProto.Query msg) throws Exception {
        Timer.Context overallContext = overallTimer.time();
        int userId = msg.getUserID();
        int k = msg.getK();
        Timer.Context predictContext = predictTimer.time();
        float[] userFeatureList;
        FeatureProto.IDs userIds = FeatureProto.IDs.newBuilder().addID(userId).build();
        FeatureProto.Features feature = featureServiceStub.getUserFeatures(userIds);
        Object[][] featureList = FeatureUtils.getFeatures(feature);
        if (featureList[0] == null) {
            throw new Exception("Can't get user feature from feature service");
        }
        userFeatureList = RecallUtils.featureObjToFloatArr(featureList[0]);
        predictContext.stop();
        Timer.Context faissContext = faissTimer.time();
        int[] candidates =
                indexService.search(IndexService.vectorToFloatArray(userFeatureList), k);
        faissContext.stop();
        RecallProto.Candidates.Builder result = RecallProto.Candidates.newBuilder();
        // TODO: length < k
        for (int i = 0; i < k; i ++) {
            result.addCandidate(candidates[i]);
        }
        overallContext.stop();
        return result.build();
    }

    public Empty addItemToIndex(RecallProto.Item msg) {
        // TODO: multi server synchronize
        System.out.printf("Index service nTotal before = %d\n", this.indexService.getNTotal());
        System.out.printf("Index service nTotal after = %d\n", this.indexService.getNTotal());
        return Empty.newBuilder().build();
    }

    private void addToIndex(int targetId, float[] vector) {
        floatArray fa = IndexService.vectorToFloatArray(vector);
        this.indexService.add(targetId, fa);
    }


}
