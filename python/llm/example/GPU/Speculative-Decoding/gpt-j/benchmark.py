#
# Copyright 2016 The BigDL Authors.
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

import torch
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import numpy as np
import pandas as pd


torch.nn.Linear.reset_parameters = lambda x: None
# seed=60
# torch.manual_seed(seed)
# np.random.seed(seed)

CHATGLM_V3_PROMPT_FORMAT = "<|user|>\n{prompt}\n<|assistant|>"

long_input = """折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。问: 请基于以上描述，分析哪些步骤做错了很大可能会导致最终折叠失败？答: """
# long_input = """上古时，女娲娘娘炼石补天，采来三万六千五百零一块顽石，只剩下一块未用，扔在大荒山青埂峰下。这块顽石经过娘娘锻炼，有了灵性，能变大变小，会自来自去。这天，一个和尚与一个道士来到青埂峰下，见这块石头洁净晶莹，只有折扇的扇坠般大小。和尚把他托在手上，说：“在你身上刻上几个字，让人们见了就知道你是个宝贝，把你带到繁荣昌盛的国家、读书识礼的豪门望族、花柳繁华富贵温柔的地方走一趟。”石头高兴万分，问：“不知刻什么字？带到哪儿？”和尚笑着说：“你先别问。将来自然明白。”说完，他把石头放在袖中，与道士飘然离去。又不知过了多少万年，有个空空道人路过大荒山无稽崖青埂峰下，见到一块巨石，上面刻着许多字，就从头到尾看了一遍。原来石上刻的是他被茫茫大士携入红尘，投胎人世间的一番经历。上面什么事情都有，只是没有朝代年月，后面还有一首诗：\n无才可去补苍天，枉入红尘若许年。\n此系身前身后事，倩谁记去作奇传?\n空空道人就把石中文字抄下来，定名为《石头记》。他因受了石上故事的影响，就改名为情僧，把《石头记》改为《情僧录》。山东的孔梅溪题为《*》。后来，曹雪芹于悼红轩中，披阅十载，增删五次，编纂成目录，分出章回，又题名为《金陵十二钗》，并题一首绝句：\n满纸荒唐言，一把辛酸泪!\n都云作者痴，谁解其中味?\n那块石头上记录的文字是这样的：\n苏州城的阊门，是人间最繁华风流的地方。阊门外有个十里街，街上有条仁清巷，巷里有座葫芦庙，庙旁住着一家官人，姓甄名费字士隐，娶妻封氏，性情贤淑。家中虽不是多富，在这一带也是第一家。他生性恬淡，不求功名，每天观花种竹、饮酒作诗，倒也神仙般快乐。美中不足的是，老夫妻年近半百，没有儿子，只一个女儿，名叫英莲，年方三岁。\n盛夏的一天，士隐在书房读书读累了，伏到几案上，矇矇眬眬地来到一个地方，就见来了一个和尚、一个道人。道人问：“你要把这蠢物带到哪里?”和尚说：“如今有一段风流公案还没了结，这些风流冤家还没投胎。趁此机会，把这石头夹带在里面，让他去经历一番。”道人问：“这些风流冤家不知起于何处?落于何方?”和尚说：“这块石头因女娲娘娘没用他，到各处游玩。这天他来到警幻仙子处，警幻仙子就命他为赤霞宫神瑛侍者。他见西方灵河岸三生石畔有绛珠仙草一株，非常可爱，就每天用甘露浇灌，使仙草脱了草木之胎，修成女儿体。仙草为报石头的浇灌之恩，在五脏中结成缠绵不尽的情意，常说：‘我若下世为人，要用一生的眼泪来报答他。’就因为这事，勾引出许多风流冤家都要下凡。我们可把这石头带到警幻仙子那里，给他挂了号，同这些情鬼下凡，了结此案。”道士说：“果然好笑，我还从未听说还泪报恩的事。你我何不趁此机会也下世度脱几个，岂不是一场功德?”\n甄士隐听到这种稀罕事，忙上前施礼，想打听明白。二仙却笑着说：“这是天机，不可泄露。”士隐一再追问，“蠢物”是什么。和尚递过一块晶莹的美玉，他接过一看，正面刻着“通灵宝玉”四个字，背面还刻着几行小字，正想细看，和尚说：“已到幻境。”就把玉夺回，与道人进入一个石牌坊。牌坊上刻“太虚幻境”，两旁是一副对联：\n假作真时真亦假，无为有处有还无。\n甄士隐想跟进去，刚一抬脚，忽听山崩地裂般一声响，忽然惊醒，原来是梦，梦中的事已忘了一半。他见乳母抱着英莲走来，伸手接过来，抱到门口看热闹。突然，街上过来一个和尚、一个道士，蓬着头，赤着脚，疯疯癫癫地说笑着走过来。和尚见他抱着女儿"""
# long_input = """患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血>细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无>异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细>胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L>；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生>理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压>、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；>病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多>见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果>：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临>床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出>血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、常用的治疗手段。答：" """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="THUDM/chatglm3-6b",
                        help='The huggingface repo id for the ChatGLM3 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--th_stop_draft', type=float, default=0.6,
                        help='draft stop probility')

    args = parser.parse_args()
    print(args)
    model_path = args.repo_id_or_model_path
    # Load model in optimized fp16 here.
    # Set `speculative=True`` to enable speculative decoding,
    # it only works when load_in_low_bit="fp16" on Intel GPU or load_in_low_bit="bf16" on latest Intel Xeon CPU
    if "chatglm" in model_path.lower():
        model = AutoModel.from_pretrained(model_path,
                                          optimize_model=True,
                                          torch_dtype=torch.float16,
                                          load_in_low_bit="fp16",
                                          speculative=True,
                                          trust_remote_code=True,
                                          use_cache=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     torch_dtype=torch.float16,
                                                     load_in_low_bit="fp16",
                                                     speculative=True,
                                                     trust_remote_code=True,
                                                     use_cache=True)
    model = model.to('xpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with torch.inference_mode():
        prompt = CHATGLM_V3_PROMPT_FORMAT.format(prompt=args.prompt)
        # prompt = args.prompt
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        input_ids_length = input_ids.shape[-1]
        print(f"input_id_len: {input_ids_length}")

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                do_sample=True,
                                th_stop_draft=args.th_stop_draft,)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        results = []
        for _ in range(10):
            print("=======================================")
            st = time.perf_counter()
            output = model.generate(input_ids,
                                    max_new_tokens=args.n_predict,
                                    do_sample=True,
                                    hf_adjust=False,
                                    max_step_draft=5,
                                    top_p=1,
                                    temperature=1,
                                    th_stop_draft=args.th_stop_draft,)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            torch.xpu.synchronize()
            end = time.perf_counter()


            print(f"Tokens generated {model.n_token_generated}")
            print(f"E2E Generation time {(end - st):.4f}s")
            print(f"First token latency {model.first_token_time:.4f}s")

            
            print(output_str)
            print(f"Final token number {model.n_token_generated}")
            print(f"Average Draft time {sum(model.draft_time)/model.n_drafted}")
            print(f"Average Verify time {sum(model.verify_time)/len(model.verify_time)}")
            print(f"Average Generation time {sum(model.generate_time)/len(model.generate_time)}")
            print(f"Generation throughput {1.0 * (model.n_token_generated - 1) / sum(model.generate_time)}")
            print(f"E2E Generation throughput without first token {1.0 * (model.n_token_generated - 1) / model.e2e_time_without_first }")
            print(f"E2E Generation throughput {1.0 * (model.n_token_generated - 1) / (end - st) }")
            print(f"Draft num {model.n_drafted}")
            print(f"Accept num {model.n_matched}")
            print(f"Draft {model.draft_num}")
            print(f"Accept {model.accept_num}")
            print(f"Iters: {len(model.draft_num)}")
            print(f"Draft len: {model.n_drafted/len(model.draft_num)}, accept len: {model.n_matched/len(model.accept_num)}")
            print(f"Accept rate: {model.n_matched/model.n_drafted}")
            results.append([args.th_stop_draft,
                            round(model.n_drafted/len(model.draft_num), 2),
                            round(model.n_matched/len(model.accept_num), 2),
                            round(model.n_matched/model.n_drafted, 2),
                            round(sum(model.verify_time)/len(model.verify_time) * 1000, 3),
                            round(sum(model.draft_time)/model.n_drafted * 1000, 3),
                            round(1.0 * (model.n_token_generated - 1) / model.e2e_time_without_first, 3)])
        df = pd.DataFrame(results, columns=["th_stop_draft", "draft_len", "accept_len", "accept_rate", "verify", "draft",
                                            "throughput"])
        print(df)
        df.to_csv(f'{args.th_stop_draft}-results.csv', index=False)
        # print(output_str)
