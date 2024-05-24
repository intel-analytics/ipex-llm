# ChatGLM3
In this directory, you will find examples on how you could run ChatGLM3 BF16 infernece with self-speculative decoding using IPEX-LLM on Intel CPUs. For illustration purposes, we utilize the [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) as a reference ChatGLM3 model.


## Example: Predict Tokens using `generate()` API
In the example [speculative.py](speculative.py), we show a basic use case for a ChatGLM3 model to predict the next N tokens using `generate()` API, with IPEX-LLM speculative decoding optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```
### 2. Configures OneAPI environment variables
```bash
source ipex-llm-init -t
export OMP_NUM_THREADS=one_socket_num
```

### 3. Run

```
(numactl -m 0 -C 0-47) python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH 
--prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM3 model to be downloaded, or the path to the huggingface checkpoint folder. It is set to `'THUDM/chatglm3-6b'` by default.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is set to `128` by default.

#### Sample Output
#### [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)
```log
[gMASK]sop <|user|>
多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命的，”吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”--丈夫回答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫芦。
三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。“在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府。
　请详细描述霍·阿·布恩蒂亚是如何是怎样从这片崭新的天地寻找金子的？吉卜赛人带来了哪些神奇的东西？
<|assistant|> 霍·阿·布恩蒂亚是一位充满激情和想象力的科学家，他对于探索未知领域和发现新知识有着极高的热情。在他的生活中，他一直致力于寻找一种能够开采地下金子的神奇工具或发明。他相信，通过科学的力量和创造力，可以找到更多的财富和资源。

吉卜赛人带来的神奇东西主要包括：

1. 磁铁：这是一种可以吸引铁器的神秘物质，可以用来进行各种有趣的实验和应用。

2. 望远镜：这是一种可以放大远距离物体的光学仪器，可以让人们观察到更远、更清晰的事物。
Tokens generated 128
E2E Generation time xx.xxxxs
First token latency xx.xxxxs
```

### 4. Accelerate with BIGDL_OPT_IPEX

BIGDL_OPT_IPEX can help to accelerate speculative decoding on chatglm3-6b, and please refer to [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/Speculative-Decoding/llama2#4-accelerate-with-ipex_opt_ipex) for a try.