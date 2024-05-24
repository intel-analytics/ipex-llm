# Baichuan2
In this directory, you will find examples on how you could run Baichuan2 BF16 inference with self-speculative decoding using IPEX-LLM on [Intel CPUs](../README.md). For illustration purposes, we utilize the [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) and [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) as reference Baichuan2 models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [speculative.py](speculative.py), we show a basic use case for a Baichuan2 model to predict the next N tokens using `generate()` API, with IPEX-LLM speculative decoding optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==2.1.0
pip install transformers==4.31.0
```
### 2. Configures high-performing processor environment variables
```bash
source ipex-llm-init -t
export OMP_NUM_THREADS=48 # you can change 48 here to #cores of one processor socket
```
### 3. Run

We recommend to use `numactl` to bind the program to a specified processor socket:

```bash
numactl -C 0-47 -m 0 python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

For example, 0-47 means bind the python program to core list 0-47 for a 48-core socket.

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Baichuan2 model (e.g. `baichuan-inc/Baichuan2-7B-Chat` and `baichuan-inc/Baichuan2-13B-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'baichuan-inc/Baichuan2-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### Sample Output
#### [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)

```log
<human>多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命的，”吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”--丈夫回答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫芦。
三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。“在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府。
　请详细描述霍·阿·布恩蒂亚是如何是怎样从这片崭新的天地寻找金子的？吉卜赛人带来了哪些神奇的东西？ <bot>霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
Tokens generated 128
E2E Generation time x.xxxxs
First token latency x.xxxxs
```

#### [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

```log
<human>多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命的，”吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。
梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”--丈夫回答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫芦。
三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。“在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府。
　请详细描述霍·阿·布恩蒂亚是如何是怎样从这片崭新的天地寻找金子的？吉卜赛人带来了哪些神奇的东西？ <bot>霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”但他不听劝阻，用自己的一匹骡子和两只山羊换下了两块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但枉费工夫。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声
Tokens generated 128
E2E Generation time x.xxxxs
First token latency x.xxxxs
```

### 4. Accelerate with BIGDL_OPT_IPEX

To accelerate speculative decoding on CPU, optionally, you can install our validated version of [IPEX 2.2.0+cpu](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu) refering to [IPEX's installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu), or by the following steps: (Other versions of IPEX may have some conflicts and can not accelerate speculative decoding correctly.)

#### 4.1 Install IPEX 2.2.0+cpu
```bash
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch==2.2.0
python -m pip install oneccl_bind_pt==2.2.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# if there is any installation problem for oneccl_binding, you can also find suitable index url at "https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/" or "https://developer.intel.com/ipex-whl-stable-cpu" according to your environment.

# Install other dependencies
python -m pip install transformers==4.36.2 transformers-stream-generator
```

#### 4.2 Run Baichuan2 Models with IPEX

After installed IPEX, **if the size of your Baichuan2 is 7B**, replace `modeling_baichuan.py` file under your model directory with `./baichaun2_7b_opt_ipex/modeling_baichuan.ipex`, like:

```bash
cp ./baichaun2_7b_opt_ipex/modeling_baichuan.ipex your_model_path/modeling_baichuan.py
```

And also replace `tokenization_baichuan.py` file under your model directory with `./baichaun2_7b_opt_ipex/tokenization_baichuan.py`.

**13B does not need the above operations, and please ignore.**

Then, you can set `BIGDL_OPT_IPEX=true` to get target model acceleration:

```bash
source ipex-llm-init -t
export BIGDL_OPT_IPEX=true
export OMP_NUM_THREADS=48 # you can change 48 here to #cores of one processor socket
numactl -C 0-47 -m 0 python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```
