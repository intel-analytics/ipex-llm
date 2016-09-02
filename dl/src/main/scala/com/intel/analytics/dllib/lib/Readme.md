Performance Comparison
=======
## 1. Layer Comparison 
In this section, we compared the performance of individual layers among WebscaleML, Torch and DL4J. Layer Comparison can make us learn about the detailed performance of each deep learning framework.

For each layer, two main procedures `forward` and `backward` were tested. The metrics we used to measure the running time is *milliseconds*. The parameters taken for each model are primarily based on the parameters used in AlexNet, CifarVGG, GoogleNet and LeNet. 

Note that the parameters of the each layer are listed in the table below:

| Model | Parameter  
|-------|------------
|BatchNormalization| Input Number: 100, Feature Dimension: 512
|BCECriterionPerform| Input Number: 100, Feature Dimension: 512
|ClassNLLCriterion| Input Number: 512, Feature Dimension: 512
|Dropout|P, Input Number: 1000, Feature Dimension: 512
|Linear|Input Size, Output Size, ForwardTimeoutMillis, BackwardTimeoutMillis, Input Number: 100
|ReLU|ip (Boolean), Input Number: 100, Feature dimension: 512
|Convolution | Input Channel, Output Channel, Kernel Width, Kernel Height, dW, dH, padding Width, padding Height, input Width, output Width, Input Number: 10, Feature dimension: 512
|MaxPooling |Kernel Width, Kernel Height, dW, dh, padding Width, padding Height, Input Number: 100, Input Height: 100, Input Width: 512, Input Channel: 3


### 1.1 Layer Performance based on single E3 CPU
The following table shows the performance of the three deeplearning framework running on E3 CPU.

| Model| Method| Parameter| WebScaleML | Torch| DL4j | IntelCaffe with MKL2017 | Caffe |
|------|-------|----------|------------|------|------|-------------------------|-------|
| ReLU| forward| true| 5.97| 20.676302909851 | -| -| - |
| ReLU| backward | true| 14.96| 31.409215927124 | - | - |- |
| ReLU| forward| false| 4.95| 31.350111961365 |2.83| 0.0| 0.0|
| ReLU| backward | false| 210.39| 33.90040397644|3.00| 0.0| 0.0|
| Convolution | forward| 3, 64, 11, 11, 4, 4, 2, 2, 224, 224| 13.13| 26.72|55.01| 9.2| 13.8|
| Convolution | forward| 64, 192, 5, 5, 1, 1, 2, 2, 25, 25| 17.14| 36.24|97.2| 0.0| 20.0|
| Convolution | forward| 191, 384, 3, 3, 1, 1, 1, 1, 12, 12| 8.64| 16.49|44.98| 857.8| 8.0|
| Convolution | forward| 384, 256, 3, 3, 1, 1, 1, 1, 6, 6| 5.14| 7.60|17.20| 0.0| 4.2|
| Convolution | forward| 256, 256, 3, 3, 1, 1, 1, 1, 3, 3| 2.00| 2.54|5.79| 4.0| 1.0|
| Convolution | forward| 3, 64, 3, 3, 1, 1, 1, 1, 224, 224| 53.89| 86.53|137.38| 21.2| 57.0|
| Convolution | forward| 64, 64, 3, 3, 1, 1, 1, 1, 110, 110| 76.32| 144.26|400.09| 46.6| 82.0|
| Convolution | forward| 64, 128, 3, 3, 1, 1, 1, 1, 54, 54| 27.64| 48.22|138.30| 27.0| 28.8|
| Convolution | forward| 128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26| 10.27| 22.16|64.02| 10.2| 12.4|
| Convolution | forward| 128, 256, 3, 3, 1, 1, 1, 1, 13, 13| 5.34| 9.93|42.14| 4.8| 5.0|
| Convolution | forward| 256, 256, 3, 3, 1, 1, 1, 1, 6, 6| 3.41| 5.25|11.37| 0.0| 3.0|
| Convolution | forward| 256, 512, 3, 3, 1, 1, 1, 1, 3, 3| 3.55| 4.87|6.52| 0.0| 3.0|
| Convolution | forward| 512, 512, 3, 3, 1, 1, 1, 1, 2, 2| 5.22| 6.96|6.28| 0.0| 4.0|
| Convolution | forward| 3, 64, 7, 7, 2, 2, 3, 3, 224, 224| 27.51| 44.63|101.30| 16.6| 26.4|
| Convolution | forward| 64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 80| 2.9| 6.65|33.73| 3.2| 2.0|
| Convolution | forward| 64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 410| 8.63| 16.09|45.52| 9.2| 8.4|
| Convolution | forward| 192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 770 | 13.36| 23.35|62.96| 14.2| 12.0|
| Convolution | forward| 576, 576, 2, 2, 2, 2, 0, 0, 4, 4| 2.63| 4.11|4.69| 67.0| 2.0|
| Convolution | backward | 3, 64, 11, 11, 4, 4, 2, 2, 224, 224| 29.09| 47.05|236.39| 48965.4| 28.8|
| Convolution | backward | 64, 192, 5, 5, 1, 1, 2, 2, 25, 25| 49.34| 62.18|325.33| 35.2| 44.2|
| Convolution | backward | 191, 384, 3, 3, 1, 1, 1, 1, 12, 12| 28.29| 38.87|147.09| 7495.0| 23.8|
| Convolution | backward | 384, 256, 3, 3, 1, 1, 1, 1, 6, 6| 18.97| 19.20|54.02| 11.8| 16.4|
| Convolution | backward | 256, 256, 3, 3, 1, 1, 1, 1, 3, 3| 8.64| 5.48|11.92| 4.0| 9.8|
| Convolution | backward | 3, 64, 3, 3, 1, 1, 1, 1, 224, 224| 59.52| 98.99|4001.30| 6922.0| 57.0|
| Convolution | backward | 64, 64, 3, 3, 1, 1, 1, 1, 110, 110| 153.35| 253.97|1575.06| 104.8| 174.6 |
| Convolution | backward | 64, 128, 3, 3, 1, 1, 1, 1, 54, 54| 53.04| 82.41|565.16| 49.6| 56.6|
| Convolution | backward | 128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26| 23.82| 36.92|222.82| 24.0| 28.4|
| Convolution | backward | 128, 256, 3, 3, 1, 1, 1, 1, 13, 13| 13.30| 19.25|84.66| 13.4| 12.8|
| Convolution | backward | 256, 256, 3, 3, 1, 1, 1, 1, 6, 6| 11.30| 11.51|34.97| 7.8| 10.4|
| Convolution | backward | 256, 512, 3, 3, 1, 1, 1, 1, 3, 3| 18.39| 11.29|19.13| 8.6| 16.2|
| Convolution | backward | 512, 512, 3, 3, 1, 1, 1, 1, 2, 2| 36.07| 33.42|19.64| 14.8| 30.0|
| Convolution | backward | 3, 64, 7, 7, 2, 2, 3, 3, 224, 224| 49.77| 87.50|675.12| 21582.4| 48.0|
| Convolution | backward | 64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 80| 5.32| 14.14|154.73| 4.2| 3.2|
| Convolution | backward | 64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 410| 17.45| 31.89|175.84| 18.2| 18.4|
| Convolution | backward | 192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 770 | 36.02| 55.83|205.79| 38.4| 32.6|
| Convolution | backward | 576, 576, 2, 2, 2, 2, 0, 0, 4, 4| 18.30| 13.95|13.01| 948.0| 19.0|
| MaxPooling| forward| 3, 3, 1, 1, 1, 1, 2, 2| 35.06| 14.21|152.16| 40.8| 13.0|
| MaxPooling| forward| 2, 2, 2, 2| 22.49| 12.57|162.81| 28.6| 6.2|
| MaxPooling| forward| 3, 3, 1, 1| 117.18| 49.49|581.15| 144.6| 19.0|
| MaxPooling| forward| 3, 3, 3, 3| 17.07| 8.56|70.61| 19.8| 6.0|
| MaxPooling| backward | 3, 3, 2, 2| 13.90| 24.93|928.22| 11.2| 13.0|
| MaxPooling| backward | 2, 2, 2, 2| 14.14| 26.30|829.19| 11.0| 13.0|
| MaxPooling| backward | 3, 3, 1, 1| 21.67| 34.72|3606.16| 19.8| 17.0|
| MaxPooling| backward | 3, 3, 3, 3| 13.68| 21.97|424.08| 9.8| 11.0|
| Linear| forward| 9216,4096| 47.92| 51.26|-| 30.0| 26.8|
| Linear| forward| 4096,4096| 21.62| 23.45|-| 13.0| 12.8|
| Linear| forward| 6400,4096| 0.98| 1.02|-| 18.0| 18.2|
| Linear| forward| 512,512| 0.32| 0.36|-| 0.0| 0.0|
| Linear| forward| 512,10| 0.03| 0.026|-| 0.0| 0.0|
| Linear| forward| 448,768| 0.41| 0.44|-| 0.0| 0.0|
| Linear| backward | 9216,4096| 84.02| 97.97|-| 55.4| 47.4|
| Linear| backward | 4096,4096| 41.69| 49.66|-| 23.2| 23.2|
| Linear| backward | 6400,4096| 1.76| 2.45|-| 32.6| 31.8|
| Linear| backward | 512,512| 0.60| 0.70|-| 0.0| 0.0|
| Linear| backward | 512,10| 0.03| 0.05|-| 0.0| 0.0|
| Linear| backward | 448,768| 0.76| 0.84|-| 0.0| 0.0|
| ClassNLLCriterion| forward| -| 0.64| 0.009|6.66| -|- |
| ClassNLLCriterion| backward |-| 0.57| 0.13|21.22|-|-|
| BatchNormalization | forward| -| 3.31| 1.21|2.60|-|- |
| BatchNormalization | backward |-| 4.44| 0.47|2.57|-| -|
| DropoutPerform| forward| 0.3| 26.12| 6.83|10.22| 0.0| 0.0|
| DropoutPerform| forward| 0.4| 31.30| 5.81|8.47| 0.0| 0.0|
| DropoutPerform| forward| 0.5| 26.89| 5.791|8.50| 0.0| 0.0|
| DropoutPerform| backward | 0.3| 13.60| 0.76|11.64| 0.0| 0.0|
| DropoutPerform| backward | 0.4| 17.34| 0.81|12.10| 0.0| 0.0|
| DropoutPerform| backward | 0.5| 13.73| 0.75|11.44| 0.0| 0.0|
| BCECriterion| forward| -| 11.40| 2.39|2.22|  - | -|
| BCECriterion| backward | -| 7.12| 0.35|10.39| - | -|
### 1.2 Layer Performance based on single E5 CPU
The following table shows the performance of the three deeplearning framework running on E5 CPU.

|              Model |   Method |                               Parameter | WebScaleML | Torch |    DL4j | IntelCaffe With MKL2017 | Caffe |
|--------------------|----------|-----------------------------------------|------------|-------|---------|-------------------------|-------|
|ReLU|forward|true|2.730|33.783|-|-|-|
|ReLU|backward|true|2.607|42.454|-|-|-|
|ReLU|forward|false|5.535|43.743|4.95|0.0|2.4|
|ReLU|backward|false|374.731|47.72|5.10|0.0|3.4|
|Convolution|forward|3, 64, 11, 11, 4, 4, 2, 2, 224, 224|6.416|25.613|77.75|2.0|6.8|
|Convolution|forward|64, 192, 5, 5, 1, 1, 2, 2, 25, 25|5.917|20.282|126.93|0.0|12.8|
|Convolution|forward|191, 384, 3, 3, 1, 1, 1, 1, 12, 12|2.988|10.983|56.23|165.2|5.2|
|Convolution|forward|384, 256, 3, 3, 1, 1, 1, 1, 6, 6|1.799|6.82|24.06|0.0|3.2|
|Convolution|forward|256, 256, 3, 3, 1, 1, 1, 1, 3, 3|0.894|2.265|7.8|0.0|1.8|
|Convolution|forward|3, 64, 3, 3, 1, 1, 1, 1, 224, 224|14.767|24.394|185.23|41.8|22.4|
|Convolution|forward|64, 64, 3, 3, 1, 1, 1, 1, 110, 110|25.876|65.744|605.04|11.6|37.2|
|Convolution|forward|64, 128, 3, 3, 1, 1, 1, 1, 54, 54|8.009|29.756|209.58|7.0|13.8|
|Convolution|forward|128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26|4.186|16.693|83.65|2.2|7.2|
|Convolution|forward|128, 256, 3, 3, 1, 1, 1, 1, 13, 13|2.103|7.858|34.54|1.6|4.0|
|Convolution|forward|256, 256, 3, 3, 1, 1, 1, 1, 6, 6|1.314|6.373|18.27|0.0|2.8|
|Convolution|forward|256, 512, 3, 3, 1, 1, 1, 1, 3, 3|1.016|3.869|10.22|0.0|3.0|
|Convolution|forward|512, 512, 3, 3, 1, 1, 1, 1, 2, 2|1.574|6.027|8.27|0.0|2.8|
|Convolution|forward|3, 64, 7, 7, 2, 2, 3, 3, 224, 224|10.123|19.523|141.10|2.4|11.0|
|Convolution|forward|64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 80|0.948|6.719|50.25|1.0|1.0|
|Convolution|forward|64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 410|2.609|16.015|58.21|2.8|5.2|
|Convolution|forward|192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 770|4.613|25.154|78.93|3.0|7.8|
|Convolution|forward|576, 576, 2, 2, 2, 2, 0, 0, 4, 4|0.692|4.187|9.13|18.0|2.0|
|Convolution|backward|3, 64, 11, 11, 4, 4, 2, 2, 224, 224|9.516|40.243|358.29|8637.4|14.0|
|Convolution|backward|64, 192, 5, 5, 1, 1, 2, 2, 25, 25|9.899|57.715|466.84|10.4|27.6|
|Convolution|backward|191, 384, 3, 3, 1, 1, 1, 1, 12, 12|6.239|37.872|190.02|1394.0|14.6|
|Convolution|backward|384, 256, 3, 3, 1, 1, 1, 1, 6, 6|4.614|29.124|90.33|3.6|13.2|
|Convolution|backward|256, 256, 3, 3, 1, 1, 1, 1, 3, 3|2.653|11.917|19.05|1.4|9.2|
|Convolution|backward|3, 64, 3, 3, 1, 1, 1, 1, 224, 224|49.727|91.365|2567.31|1311.4|35.0|
|Convolution|backward|64, 64, 3, 3, 1, 1, 1, 1, 110, 110|41.542|148.739|2169.82|32.8|78.8|
|Convolution|backward|64, 128, 3, 3, 1, 1, 1, 1, 54, 54|12.095|78.762|728.53|13.2|27.8|
|Convolution|backward|128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26|6.328|46.212|356.16|6.0|14.4|
|Convolution|backward|128, 256, 3, 3, 1, 1, 1, 1, 13, 13|3.332|35.537|120.98|3.8|9.6|
|Convolution|backward|256, 256, 3, 3, 1, 1, 1, 1, 6, 6|2.811|26.69|48.34|2.4|10.8|
|Convolution|backward|256, 512, 3, 3, 1, 1, 1, 1, 3, 3|3.579|32.082|32.78|2.2|15.4|
|Convolution|backward|512, 512, 3, 3, 1, 1, 1, 1, 2, 2|8.304|19.334|28.17|4.8|25.4|
|Convolution|backward|3, 64, 7, 7, 2, 2, 3, 3, 224, 224|49.543|66.905|782.76|4269.6|22.2|
|Convolution|backward|64, 64, 1, 1, 1, 1, 0, 0, 54, 54, 80|2.590|22.461|223.85|1.0|2.6|
|Convolution|backward|64, 192, 3, 3, 1, 1, 1, 1, 27, 27, 410|6.537|45.453|224.57|5.4|11.2|
|Convolution|backward|192, 576, 3, 3, 1, 1, 1, 1, 12, 12, 770|7.572|49.048|269.96|9.0|21.6|
|Convolution|backward|576, 576, 2, 2, 2, 2, 0, 0, 4, 4|5.408|21.023|22.52|233.4|15.2|
|MaxPooling|forward|3, 3, 1, 1, 1, 1, 2, 2|9.126|13.115|429.37|10.8|4.0|
|MaxPooling|forward|2, 2, 2, 2|6.063|3.719|312.18|7.4|2.4|
|MaxPooling|forward|3, 3, 1, 1|28.680|15.285|1728.94|39.0|5.6|
|MaxPooling|forward|3, 3, 3, 3|4.357|3.694|192.01|5.4|3.4|
|MaxPooling|backward|3, 3, 1, 1, 1, 1, 2, 2|9.962|21.503|1342.94|6.2|2.6|
|MaxPooling|backward|2, 2, 2, 2|12.318|21.681|1145.90|4.0|2.8|
|MaxPooling|backward|3, 3, 1, 1|11.392|26.324|5323.16|7.0|6.2|
|MaxPooling|backward|3, 3, 3, 3|9.189|21.296|625.67|9.8|2.0|
|Linear|forward|9216,4096|15.188|40.368|-|13.4|13.6|
|Linear|forward|4096,4096|8.522|17.895|-|7.0|7.0|
|Linear|forward|6400,4096|0.586|0.648|-|9.8|10.0|
|Linear|forward|512,512|0.224|0.329|-|0.0|0.0|
|Linear|forward|512,10|0.070|0.051|-|0.0|0.0|
|Linear|forward|448,768|0.266|0.606|-|0.0|0.0|
|Linear|backward|9216,4096|24.327|73.516|-|25.4|26.6|
|Linear|backward|4096,4096|9.540|33.968|-|12.2|12.2|
|Linear|backward|6400,4096|0.592|21.154|-|15.6|15.6|
|Linear|backward|512,512|0.316|3.991|-|0.0|0.0|
|Linear|backward|512,10|0.104|0.166|-|0.0|0.0|
|Linear|backward|448,768|0.365|4.921|-|0.0|0.0|
|ClassNLLCriterion|forward|-|0.525|0.019|8.45|-|-|
|ClassNLLCriterion|backward|-|0.651|0.301|30.63|-|-|
|BatchNormalization|forward|-|1.346|6.013|2.60|-|-|
|BatchNormalization|backward|-|4.313|6.259|2.57|-|-|
|DropoutPerform|forward|0.3|34.639|20.862|14.60|3.4|0.0|
|DropoutPerform|forward|0.4|42.931|12.945|13.26|0.8|1.6|
|DropoutPerform|forward|0.5|33.978|8.187|18.45|0.0|2.6|
|DropoutPerform|backward|0.3|16.216|0.984|23.32|0.0|0.0|
|DropoutPerform|backward|0.4|16.097|0.948|23.44|0.0|0.0|
|DropoutPerform|backward|0.5|16.072|0.929|25.49|0.0|0.6|
|BCECriterion|forward|-|15.715|7.022|4.82|-|-|
|BCECriterion | backward |-|15.093|0.855| 20.43 | -| - |
