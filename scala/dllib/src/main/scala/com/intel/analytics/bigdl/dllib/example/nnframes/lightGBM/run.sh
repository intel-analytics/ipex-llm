export SPARK_HOME=/Users/guoqiong/intelWork/tools/spark/spark-3.1.2-bin-hadoop2.7
export BIGDL_HOME=/Users/guoqiong/intelWork/git/BigDL/dist
#${BIGDL_HOME}/bin/spark-submit-with-dllib.sh \
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --class com.intel.analytics.bigdl.dllib.example.nnframes.lightGBM.LgbmClassifierTrain \
  /Users/guoqiong/intelWork/git/BigDL/dist/lib/bigdl-dllib-spark_3.1.2-2.2.0-SNAPSHOT-jar-with-dependencies.jar \
  --inputPath /Users/guoqiong/intelWork/data/tweet/xgb_processed \
  --numIterations 100 \
  --modelSavePath /tmp/lgbm/scala/classifier
