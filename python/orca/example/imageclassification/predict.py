#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from optparse import OptionParser
from zoo.common.nncontext import init_nncontext
from zoo.models.image.imageclassification import *


def predict(model_path, img_path, topN, partition_num):
    print("ImageClassification prediction")
    print("Model Path %s" % model_path)
    print("Image Path %s" % img_path)
    print("Top N : %d" % topN)
    imc = ImageClassifier.load_model(model_path)
    image_set = ImageSet.read(img_path, sc, partition_num)
    output = imc.predict_image_set(image_set)
    labelMap = imc.get_config().label_map()
    predicts = output.get_predict().collect()
    for predict in predicts:
        (uri, probs) = predict
        sortedProbs = [(prob, index) for index, prob in enumerate(probs[0])]
        sortedProbs.sort()
        print("Image : %s, top %d prediction result" % (uri, topN))
        for i in range(topN):
            print("\t%s, %f" % (labelMap[sortedProbs[999 - i][1]], sortedProbs[999 - i][0]))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--folder", type=str, dest="img_path", default=".",
                      help="Path where the images are stored")
    parser.add_option("--model", type=str, dest="model_path", default="",
                      help="Path where the model is stored")
    parser.add_option("--topN", type=int, dest="topN", default=1, help="top N number")
    parser.add_option("--partition_num", type=int, dest="partition_num", default=4,
                      help="The number of partitions")
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("Image Classification Example")

    predict(options.model_path, options.img_path, options.topN, options.partition_num)
    print("finished...")
    sc.stop()
