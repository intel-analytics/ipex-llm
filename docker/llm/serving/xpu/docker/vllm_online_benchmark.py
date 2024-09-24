import requests
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
from tqdm import tqdm
import json
import random
import sys

if len(sys.argv) < 3:
    print("Usage: python bench.py <model> <max_seq> [input_length] [output_length]")
    sys.exit(1)

print("running bench.py")
model_name = sys.argv[1]
print("model_name: " + str(model_name))
max_seq = sys.argv[2]
print("max_seq: " + str(max_seq))

input_length = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
print("input_length: " + str(input_length))
output_length = int(sys.argv[4]) if len(sys.argv) > 4 else 512
print("output_length: " + str(output_length))

PROMPT_32 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

PROMPT_128 = "In a distant future, humanity has expanded across the galaxy, establishing colonies on numerous planets. The interstellar community thrives under the guidance of the United Galactic Federation, which ensures peace and prosperity. However, a new threat emerges from the unknown regions of space, challenging the stability and security of the galaxy. Brave explorers and seasoned warriors must unite to uncover the secrets of this mysterious force and protect the future of all sentient beings.  Please continue the above story as long as possible, preferably more than 1000 tokens."

PROMPT_1024 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. However, her parents were always telling her to stay close to home, to be careful, and to avoid any danger. But the little girl was stubborn, and she wanted to see what was on the other side of the mountain. So she sneaked out of the house one night, leaving a note for her parents, and set off on her journey. As she climbed the mountain, the little girl felt a sense of excitement and wonder. She had never been this far away from home before, and she couldnt wait to see what she would find on the other side. She climbed higher and higher, her lungs burning from the thin air, until she finally reached the top of the mountain. And there, she found a beautiful meadow filled with wildflowers and a sparkling stream. The little girl danced and played in the meadow, feeling free and alive. She knew she had to return home eventually, but for now, she was content to enjoy her adventure. As the sun began to set, the little girl reluctantly made her way back down the mountain, but she knew that she would never forget her adventure and the joy of discovering something new and exciting. And whenever she felt scared or unsure, she would remember the thrill of climbing the mountain and the beauty of the meadow on the other side, and she would know that she could face any challenge that came her way, with courage and determination. She carried the memories of her journey in her heart, a constant reminder of the strength she possessed. The little girl returned home to her worried parents, who had discovered her note and anxiously awaited her arrival. They scolded her for disobeying their instructions and venturing into the unknown. But as they looked into her sparkling eyes and saw the glow on her face, their anger softened. They realized that their little girl had grown, that she had experienced something extraordinary. The little girl shared her tales of the mountain and the meadow with her parents, painting vivid pictures with her words. She spoke of the breathtaking view from the mountaintop, where the world seemed to stretch endlessly before her. She described the delicate petals of the wildflowers, vibrant hues that danced in the gentle breeze. And she recounted the soothing melody of the sparkling stream, its waters reflecting the golden rays of the setting sun. Her parents listened intently, captivated by her story. They realized that their daughter had discovered a part of herself on that journey—a spirit of curiosity and a thirst for exploration. They saw that she had learned valuable lessons about independence, resilience, and the beauty that lies beyond ones comfort zone. From that day forward, the little girls parents encouraged her to pursue her dreams and embrace new experiences. They understood that while there were risks in the world, there were also rewards waiting to be discovered. They supported her as she continued to embark on adventures, always reminding her to stay safe but never stifling her spirit. As the years passed, the little girl grew into a remarkable woman, fearlessly exploring the world and making a difference wherever she went. The lessons she had learned on that fateful journey stayed with her, guiding her through challenges and inspiring her to live life to the fullest. And so, the once timid little girl became a symbol of courage and resilience, a reminder to all who knew her that the greatest joys in life often lie just beyond the mountains we fear to climb. Her story spread far and wide, inspiring others to embrace their own journeys and discover the wonders that awaited them. In the end, the little girls adventure became a timeless tale, passed down through generations, reminding us all that sometimes, the greatest rewards come to those who dare to step into the unknown and follow their hearts. With each passing day, the little girls story continued to inspire countless individuals, igniting a spark within their souls and encouraging them to embark on their own extraordinary adventures. The tale of her bravery and determination resonated deeply with people from all walks of life, reminding them of the limitless possibilities that awaited them beyond the boundaries of their comfort zones. People marveled at the little girls unwavering spirit and her unwavering belief in the power of dreams. They saw themselves reflected in her journey, finding solace in the knowledge that they too could overcome their fears and pursue their passions. The little girl\'s story became a beacon of hope, a testament to the human spirit"

PROMPT_2048 = "“You’re an idiot,” she said.\nI smiled and leaned back in the chair, looking at her over my glasses. “No, I’m not.”\n“If you were smart you would have learned to dance years ago. You’ve got two left feet.” She held up both of her hands with four fingers extended then made a circular motion that looked like an airplane.\nI leaned forward and put my glasses on the table in front of me, reaching for her hands as I did so, grabbing them before they could leave mine. “The next time you do something like this, call me. The phone number is right here,” I said as I pointed at a piece of paper under a stack of papers on my desk.\n“Fine,” she huffed and turned to leave the room. But she stopped at the doorway when she saw the bookshelves that lined one wall. “What are these for?” She stepped closer, tilting her head back and forth as she looked up. The shelves were three stories high with stacks of books on every level.\n“Books.” I smiled again. “I have a lot of books.”\nShe didn’t respond to that so I continued: “And there are more in the basement.”\n“But you can’t move them all here, right? This place is just too small for all those books. Maybe we should look for a bigger office building.” She looked back at me but said nothing as she took another few steps towards the door and then stopped again when she saw my grandfather clock on the wall.\n“And this?” she pointed to the clock, which had been in the family for over seventy years. “It’s just a clock isn’t it?”\nI laughed. “You can say that, but I know better.” It was then that I told her my grandfather’s story. He made that clock, and it was his favorite possession. When he died she inherited the clock; or at least she thought she did. After a few weeks of trying to sell it on eBay, she gave up because no one would pay what she felt it was worth.\n“You should have had an auction,” she suggested, leaning in towards me again. “Then maybe you could get more for it.”\n“No,” I shook my head. “I don’t want to sell the clock.”\nShe smiled, but this time it didn’t reach her eyes. She took a step back and looked at me again, not saying anything, just staring. The only sound was the ticking of the grandfather clock in the background as she waited for my next words.\n“My grandfather made this clock. He did everything by hand.” I could see that she had no idea what to say or do so I continued: “It’s his favorite possession, and it means more to me than anything else he ever owned. So, if you want the books, you can have them…” I looked at her face for just a second before continuing, “but you won’t take the clock.”\nShe finally responded with: “But what about the money?” She looked around again and said, “I think we could make more selling these books than you would get from all of them. You must have thousands of books here!”\nI took another step forward and put my hand on her shoulder as I spoke to her in a very low voice. “You’ve got it all wrong,” I told her. “There are only two or three hundred books. I’m not looking for money – I’m looking for someone who understands how important this clock is.”\n“How much do you want for the books?” she asked, still staring at me intently as she waited for my answer.\n“Forget about the money,” I said again. “If you really want to buy them, we can take our time and talk more later. But if you just want their value in paperbacks, that’s not what they’re worth.” She still seemed confused by everything I had said so far, so I tried to simplify my words as much as possible: “The books are mine; the clock is my grandfather’s. These books have been passed down through several generations of our family and are irreplaceable. Do you understand?”\n“I guess not,” she answered as she walked away from me, still looking at me but not saying a word. She took two more steps before turning around to say one last thing: “Well, good luck with the books, then.” With that, she went back into her house and out of sight, still walking without talking.\nAfter a few minutes, I slowly walked back toward my grandfather’s home. As I got closer, I could see the roof in the distance; the white crosses on the top of it were hard to miss. It seemed as if the entire town had gathered around there at that moment – people were all over the road around us, watching the commotion and chattering about what was going on.\nWhen my grandfather first saw me, he looked up from his chair with a smile on his face: “There you are.” He looked down at his hands, then back toward me as I walked forward to talk to him for the first time in years: “It’s been too long since we last spoke; it’s good to see you again.”\n“And you,” I said. Then, looking past my grandfather and directly into the face of the man who was sitting next to him (my mother’s father), I said, “I see he got your clock back for you, too. How is he?” My grandfather smiled as he looked up at me again:\n“He’s fine,” he answered, still smiling as he watched my mother’s family and mine chat with one another in the middle of all these people – a situation that I had never seen before. “Come on inside.” He stood up from his chair to do just that; my mom and her sister were already walking out of the building. “I have things for you.”\nMy grandfather led us inside, down some steps where he used to serve as the pastor in his church; there was a big room full of chairs at the bottom with pictures on the wall – all kinds of pictures, from when my family first started coming here to visit and other pictures we took while staying here over the years. All these photographs were all around us as I followed my grandfather through the building:\n“My house is just up the street,” he said. He stopped at a picture on the wall that was taken in the summer when we came to visit, smiling as he looked toward it with his arms folded – the picture was of him and his wife and two of their daughters, all standing together by one of the trees outside; there were other pictures around this one, some from much earlier than when my grandfather first started serving here. “We used to sit in a booth in that restaurant right over there – you remember?” I nodded as we went past it.\nMy grandfather stopped at another picture on the wall: it was of him and his wife with two other families, all sitting around a table together, smiling. He looked down at this one for a moment; then he said, “We used to do things like this every year, when we came to visit.” It was an older picture than the last one my grandfather had stopped in front of; I didn’t know it before but now I realized how much he has aged.\nMy grandparents have lived together for many years. They used to live in a house right next door, so they could walk over whenever they wanted; that is what they have done here all these years – as my grandfather said, “we’ve come here every summer since I was eleven.” But he and his wife are getting old now. He isn’t able to walk much anymore, but it makes him happy when he does: “My health has not been good lately,” he said.\n“You will never have a better time in your life than this one right now; you will never be as happy as you are now.” And for the first time since I have known him – since I was very little and started coming here every summer – my grandfather smiled at me, his eyes sparkling with excitement.\n“I know,” I said. “That’s why I’m really looking forward to it. It will be a lot of fun.” Then he turned back to the picture again; “See this?” he asked, pointing. “I remember that day, all sixteen of us there together. I was eleven then – my dad had taken me and my brother for our first trip away from home – and that was when we used to go to the cottage.” He stared at it for a while longer; he had tears in his eyes. “I loved this picture,” he said, turning it over again with one hand so I could see the back of it.\n“This is my best memory,” he explained. “It was taken on my birthday. That’s what makes me happiest.” He pointed to a man who had a pipe in his mouth. “That’s my uncle,” he said. “He gave all of us kids cigars for our birthdays, and we used to take turns lighting them – then everyone would sit around outside in the sunshine and smoke together like that. It was such a good time.” Then he held up his hand, as if to say, that’s enough now; and he went on, “Anyway, I don’"

CHINESE_PROMPT = """
多年以后，奥雷连诺上校站在行刑队面前，准会想起父亲带他去参观冰块的那个遥远的下午。当时，马孔多是个二十户人家的村庄，一座座土房都盖在河岸上，河水清澈，沿着遍布石头的河床流去，河里的石头光滑、洁白，活象史前的巨蛋。

这块天地还是新开辟的，许多东西都叫不出名字，不得不用手指指点点。每年三月，衣衫褴楼的吉卜赛人都要在村边搭起帐篷，在笛鼓的喧嚣声中，向马孔多的居 民介绍科学家的最新发明。他们首先带来的是磁铁。一个身躯高大的吉卜赛人，自称梅尔加德斯，满脸络腮胡子，手指瘦得象鸟的爪子，向观众出色地表演了他所谓 的马其顿炼金术士创造的世界第八奇迹。他手里拿着两大块磁铁，从一座农舍走到另一座农舍，大家都惊异地看见，铁锅、铁盆、铁钳、铁炉都从原地倒下，木板上 的钉子和螺丝嘎吱嘎吱地拼命想挣脱出来，甚至那些早就丢失的东西也从找过多次的地方兀然出现，乱七八糟地跟在梅尔加德斯的魔铁后面。“东西也是有生命 的，”

吉卜赛人用刺耳的声调说，“只消唤起它们的灵性。”霍·阿·布恩蒂亚狂热的想象力经常超过大自然的创造力，甚至越过奇迹和魔力的限度，他认为这种暂时无用的科学发明可以用来开采地下的金子。

梅尔加德斯是个诚实的人，他告诫说：“磁铁干这个却不行。”可是霍·阿·布恩蒂亚当时还不相信吉卜赛人的诚实，因此用自己的一匹骡子和两只山羊换下了两 块磁铁。这些家畜是他的妻子打算用来振兴破败的家业的，她试图阻止他，但是枉费工夫。“咱们很快就会有足够的金子，用来铺家里的地都有余啦。”——丈夫回 答她。在好儿个月里，霍·阿·布恩蒂亚都顽强地努力履行自己的诺言。他带者两块磁铁，大声地不断念着梅尔加德斯教他的咒语，勘察了周围整个地区的一寸寸土 地，甚至河床。但他掘出的唯一的东西，是十五世纪的一件铠甲，它的各部分都已锈得连在一起，用手一敲，皑甲里面就发出空洞的回声，仿佛一只塞满石子的大葫 芦。

三月间，吉卜赛人又来了。现在他们带来的是一架望远镜和一只大小似鼓的放大镜，说是阿姆斯特丹犹太人的最新发明。他们把望远镜安在 帐篷门口，而让一个吉卜赛女人站在村子尽头。花五个里亚尔，任何人都可从望远镜里看见那个仿佛近在飓尺的吉卜赛女人。“科学缩短了距离。”梅尔加德斯说。 “在短时期内，人们足不出户，就可看到世界上任何地方发生的事儿。”在一个炎热的晌午，吉卜赛人用放大镜作了一次惊人的表演：他们在街道中间放了一堆干 草，借太阳光的焦点让干草燃了起来。磁铁的试验失败之后，霍·阿·布恩蒂亚还不甘心，马上又产生了利用这个发明作为作战武器的念头。梅尔加德斯又想劝阻 他，但他终于同意用两块磁铁和三枚殖民地时期的金币交换放大镜。乌苏娜伤心得流了泪。这些钱是从一盒金鱼卫拿出来的，那盒金币由她父亲一生节衣缩食积攒下 来，她一直把它埋藏在自个儿床下，想在适当的时刻使用。霍·阿·布恩蒂亚无心抚慰妻子，他以科学家的忘我精神，甚至冒着生命危险，一头扎进了作战试验。他 想证明用放大镜对付敌军的效力，就力阳光的焦点射到自己身上，因此受到灼伤，伤处溃烂，很久都没痊愈。这种危险的发明把他的妻子吓坏了，但他不顾妻子的反 对，有一次甚至准备点燃自己的房子。霍·阿·布恩蒂亚待在自己的房间里总是一连几个小时，计算新式武器的战略威力，甚至编写了一份使用这种武器的《指 南》，阐述异常清楚，论据确凿有力。他把这份《指南》连同许多试验说明和几幅图解，请一个信使送给政府；

这个信使翻过山岭，涉过茫茫苍 苍的沼地，游过汹涌澎湃的河流，冒着死于野兽和疫病的危阶，终于到了一条驿道。当时前往首都尽管是不大可能的，霍·阿·布恩蒂亚还是答应，只要政府一声令 下，他就去向军事长官们实际表演他的发明，甚至亲自训练他们掌握太阳战的复杂技术。他等待答复等了几年。最后等得厌烦了，他就为这新的失败埋怨梅尔加德 斯，于是吉卜赛人令人信服地证明了自己的诚实：他归还了金币，换回了放大镜，并且给了霍·阿·布恩蒂亚几幅葡萄牙航海图和各种航海仪器。梅尔加德斯亲手记 下了修道士赫尔曼着作的简要说明，把记录留给霍·阿·布恩蒂亚，让他知道如何使用观象仪、罗盘和六分仪。在雨季的漫长月份里，霍·阿·布恩蒂亚部把自己关 在宅子深处的小房间里，不让别人打扰他的试验。他完全抛弃了家务，整夜整夜呆在院子里观察星星的运行；为了找到子午线的确定方法，他差点儿中了暑。他完全 掌握了自己的仪器以后，就设想出了空间的概念，今后，他不走出自己的房间，就能在陌生的海洋上航行，考察荒无人烟的土地，并且跟珍禽异兽打上交道了。正是 从这个时候起，他养成了自言自语的习惯，在屋子里踱来踱去，对谁也不答理，而乌苏娜和孩子们却在菜园里忙得喘不过气来，照料香蕉和海芋、木薯和山药、南瓜 和茄子。可是不久，霍·阿·布恩蒂亚紧张的工作突然停辍，他陷入一种种魄颠倒的状态。好几天，他仿佛中了魔，总是低声地嘟嚷什么，并为自己反复斟酌的各种 假设感到吃惊，自己都不相信。最后，在十二月里的一个星期、吃午饭的时候，他忽然一下子摆脱了恼人的疑虑。孩子们至死部记得，由于长期熬夜和冥思苦想而变 得精疲力竭的父亲，如何洋洋得意地向他们宣布自己的发现：

“地球是圆的，象橙子。”

乌苏娜失去了耐心，“如果你想发 癫，你就自个几发吧！”她嚷叫起来，“别给孩子们的脑瓜里灌输古卜赛人的胡思乱想。”霍·阿·布恩蒂亚一动不动，妻子气得把观象仪摔到地上，也没有吓倒 他。他另做了一个观象仪，并且把村里的一些男人召到自己的小房间里，根据在场的人椎也不明白的理论，向他们证明说，如果一直往东航行，就能回到出发的地 点。马孔多的人以为霍·阿·布恩蒂亚疯了，可兄梅尔加德斯回来之后，马上消除了大家的疑虑。他大声地赞扬霍·阿·布恩蒂亚的智慧：光靠现象仪的探测就证实 了一种理论，这种理论虽是马孔多的居民宜今还不知道的，但实际上早就证实了；梅尔加德斯为了表示钦佩，赠给霍·阿·布恩蒂亚一套东西——炼金试验室设备， 这对全村的未来将会产生深远的影响。
十六世纪，海盗弗兰西斯·德拉克围攻列奥阿察的时候，乌苏娜。伊古阿兰的曾祖母被当当的警钟声和隆隆的炮击声吓坏了，由于神经紧张，竞一屁股坐 在生了火的炉子上。因此，曾祖母受了严重的的伤，再也无法过夫妻生活。她只能用半个屁股坐着，而且只能坐在软垫子上，步态显然也是不雅观的；所以，她就不 愿在旁人面前走路了。她认为自己身上有一股焦糊味儿，也就拒绝跟任何人交往。她经常在院子里过夜，一直呆到天亮，不敢走进卧室去睡觉：因为她老是梦见英国 人带着恶狗爬进窗子，用烧红的铁器无耻地刑讯她。她给丈夫生了两个儿子；她的丈夫是亚拉冈的商人，把自己的一半钱财都用来医治妻子，希望尽量减轻她的痛 苦。最后，他盘掉自己的店铺，带者一家人远远地离开海滨，到了印第安人的一个村庄，村庄是在山脚下，他在那儿为妻子盖了一座没有窗子的住房，免得她梦中的 海盗钻进屋子。

在这荒僻的村子里，早就有个两班牙人的后裔，叫做霍塞·阿卡蒂奥·布恩蒂亚，他是栽种烟草的；乌苏娜的曾祖父和他一起经 营这桩有利可图的事业，短时期内两人都建立了很好的家业。多少年过去了，西班牙后裔的曾孙儿和亚拉冈人的曾孙女结了婚。每当大夫的荒唐行为使乌苏娜生气的 时候，她就一下子跳过世事纷繁的三百年，咒骂弗兰西斯·德拉克围攻列奥阿察的那个日子。不过，她这么做，只是为了减轻心中的痛苦；实际上，把她跟他终生连 接在一起的，是比爱情更牢固的关系：共同的良心谴责。乌苏娜和丈夫是表兄妹，他俩是在古老的村子里一块儿长大的，由于沮祖辈辈的垦殖，这个村庄已经成了今 省最好的一个。尽管他俩之间的婚姻是他俩刚刚出世就能预见到的，然而两个年轻人表示结婚愿望的时候，双方的家长都反对。几百年来，两族的人是杂配的，他们 生怕这两个健全的后代可能丢脸地生出一只蜥蜴。这样可怕的事已经发牛过一次。乌苏娜的婶婶嫁给霍·阿·布恩蒂亚的叔叔，生下了一个儿子：这个儿子一辈子部 穿着肥大的灯笼裤，活到四十二岁还没结婚就流血而死，因为他生下来就长着一条尾巴——尖端有一撮毛的螺旋形软骨。这种名副其实的猪尾巴是他不愿让任何一个 女人看见的，最终要了他的命，因为一个熟识的屠夫按照他的要求，用切肉刀把它割掉了。十九岁的霍·阿·布恩蒂亚无忧无虑地用一句话结束了争论：“我可不在 乎生出猪崽子，只要它们会说话就行。”于是他俩在花炮声中举行了婚礼铜管乐队，一连闹腾了三个昼夜。在这以后，年轻夫妇本来可以幸福地生活，可是乌苏娜的 母亲却对未来的后代作出不大吉利的预言，借以吓唬自己的女儿，甚至怂恿女儿拒绝按照章法跟他结合。她知道大夫是个力大、刚强的人，担心他在她睡着时强迫 她，所以，她在上床之前，都穿上母亲拿厚帆布给她缝成的一条衬裤；衬裤是用交叉的皮带系住的，前面用一个大铁扣扣紧。夫妇俩就这样过了若干月。白天，他照 料自己的斗鸡，她就和母亲一块儿在刺染上绣花。夜晚，年轻夫妇却陷入了烦恼而激烈的斗争，这种斗争逐渐代替了爱情的安慰。可是，机灵的邻人立即觉得情况不 妙，而且村中传说，乌苏娜出嫁一年以后依然是个处女，因为丈大有点儿毛病。霍·阿·布恩蒂亚是最后听到这个谣言的。

“乌苏娜，你听人家在说什么啦，”他向妻子平静他说。

“让他们去嚼舌头吧，”她回答。“咱们知道那不是真的。”

他们的生活又这样过了半年，直到那个倒霉的星期天，霍·阿·布恩蒂亚的公鸡战胜了普鲁登希奥·阿吉廖尔的公鸡。输了的普鲁登希奥·阿吉廖尔，一见鸡血就气得发疯，故意离开霍·阿·布恩蒂亚远一点儿，想让斗鸡棚里的人都能听到他的话。

“恭喜你呀！”他叫道。“也许你的这只公鸡能够帮你老婆的忙。咱们瞧吧！”

霍·阿·布恩蒂亚不动声色地从地上拎起自己的公鸡。“我马上就来，”他对大家说，然后转向普鲁登希奥，阿吉廖尔：

“你回去拿武器吧，我准备杀死你。”

过了十分钟，他就拿着一枝粗大的标枪回来了，这标枪还是他祖父的。斗鸡棚门口拥聚了几乎半个村子的人，普鲁登希奥·阿吉廖尔正在那儿等候。他还来不及自 卫，霍·阿·布恩蒂亚的标枪就击中了他的咽喉，标枪是猛力掷出的，非常准确；由于这种无可指摘的准确，霍塞·奥雷连诺·布恩蒂亚（注：布恩蒂亚的祖父）

从前曾消灭了全区所有的豹子。夜晚在斗鸡棚里，亲友们守在死者棺材旁边的时候，霍·阿·布恩蒂业走进自己的卧室，看见妻子正在穿她的“贞节裤”。他拿标枪对准她，命令道：“脱掉！”乌苏娜并不怀疑丈夫的决心。“出了事，你负责，”

她警告说。霍·阿·布恩蒂亚把标枪插入泥地。

“你生下蜥蜴，咱们就抚养蜥蜴，”他说。“可是村里再也不会有人由于你的过错而被杀死了。”

这是一个美妙的六月的夜晚，月光皎洁，凉爽宜人。他俩通古未睡，在床上折腾，根本没去理会穿过卧室的轻风，风儿带来了普鲁登希奥·阿吉廖尔亲人的哭声。

人们把这桩事情说成是光荣的决斗，可是两夫妇却感到了良心的谴责。有一天夜里，乌苏娜还没睡觉，出去喝水，在院子里的大土罐旁边看见了普鲁登希奥·阿吉 廖尔。他脸色死白、十分悲伤，试图用一块麻屑堵住喉部正在流血的伤口。看见死人，乌苏娜感到的不是恐惧，而是怜悯。她回到卧室里，把这件怪事告诉了丈夫， 可是丈夫并不重视她的话。“死人是不会走出坟墓的，”他说。“这不过是咱们受到良心的责备。”过了两夜，乌苏娜在浴室里遇见普鲁登希奥·阿吉廖尔——他正 在用麻屑擦洗脖子上的凝血。另一个夜晚，她发现他在雨下徘徊。霍·阿·布恩蒂亚讨厌妻子的幻象，就带着标枪到院子里去。死人照旧悲伤地立在那儿。

“滚开！”霍·阿·布恩蒂亚向他吆喝。“你回来多少次，我就要打死你多少次。”
皮拉·苔列娜的儿子出世以后两个星期，祖父和祖母把他接到了家里。乌苏娜是勉强收留这小孩儿的，因为她又没拗过丈大的固执脾气；想让布恩蒂亚家 的后代听天由命，是他不能容忍的。但她提出了个条件：决不让孩子知道自己的真正出身。孩子也取名霍·阿卡蒂奥，可是为了避免混淆不清，大家渐渐地只管他叫 阿卡蒂奥了。这时，马孔多事业兴旺，布恩蒂亚家中一片忙碌，孩子们的照顾就降到了次要地位，负责照拂他们的是古阿吉洛部族的一个印第安女人，她是和弟弟一 块儿来到马孔多的，借以逃避他们家乡已经猖獗几年的致命传染病——失眠症。姐弟俩都是驯良、勤劳的人，乌苏娜雇用他们帮她做些家务。所以，阿卡蒂奥和阿玛 兰塔首先说的是古阿吉洛语，然后才说西班牙语，而且学会喝晰蜴汤、吃蜘蛛蛋，可是乌苏娜根本没有发现这一点，因她制作获利不小的糖鸟糖兽太忙了。马孔多完 全改变了面貌。乌苏娜带到这儿来的那些人，到处宣扬马孔多地理位置很好、周围土地肥沃，以致这个小小的村庄很快变戍了一个热闹的市镇，开设了商店和手工业 作坊，修筑了永久的商道，第一批阿拉伯人沿着这条道路来到了这儿，他们穿着宽大的裤子，戴着耳环，用玻璃珠项链交换鹦鹉。霍·阿·布恩蒂亚没有一分钟的休 息。他对周围的现实生活入了迷，觉得这种生活比他想象的大于世界奇妙得多，于是失去了对炼金试验的任何兴趣，把月复一月变来变去的东西搁在一边，重新成了 一个有事业心的、精力充沛的人了，从前，在哪儿铺设街道，在哪儿建筑新的房舍，都是由他决定的，他不让任何人享有别人没有的特权。新来的居民也十分尊敬 他，甚至请他划分土地。没有征得他的同意，就不放下一块基石，也不砌上一道墙垣。玩杂技的吉卜赛人回来的时候，他们的活动游艺场现在变成了一个大赌场，受 到热烈的欢迎。因为大家都希望霍·阿卡蒂奥也跟他们一块儿回来。但是霍·阿卡蒂奥并没有回来，那个“蛇人”也没有跟他们在一起，照乌苏娜看来，那个“蛇人 是唯”一知道能在哪儿找到她的儿子的；因此，他们不让吉卜赛人在马孔多停留，甚至不准他们以后再来这儿：现在他们已经认为吉卜赛人是贪婪佚的化身了。然而 霍·阿·布恩蒂亚却认为，古老的梅尔加德斯部族用它多年的知识和奇异的发明大大促进了马孔多的发展，这里的人永远都会张开双臂欢迎他们。可是，照流浪汉们 的说法，梅尔加德斯部族已从地面上消失了，因为他们竟敢超越人类知识的限度。

霍·阿·布恩蒂亚至少暂时摆脱了幻想的折磨以后，在短时期 内就有条不紊地整顿好了全镇的劳动生活；平静的空气是霍·阿·布恩蒂亚有一次自己破坏的，当时他放走了马孔多建立之初用响亮的叫声报告时刻的鸟儿，而给每 一座房子安了一个音乐钟。这些雕木作成的漂亮的钟，是用鹦鹉向阿拉伯人换来的，霍·阿·布恩蒂亚把它们拨得挺准，每过半小时，它们就奏出同一支华尔兹舞曲 的几节曲于让全镇高兴一次，——每一次都是几节新的曲于，到了晌午时分，所有的钟一齐奏出整支华尔兹舞曲，一点几也不走调。在街上栽种杏树，代替槐树，也 是霍·阿·布恩蒂亚的主意，而且他还发明了一种使这些杏树永远活着的办法（这个办法他至死没有透露）。过了多年，马孔多建筑了一座座锌顶木房的时候，在它 最老的街道上仍然挺立着一棵棵杏树，树枝折断，布满尘埃，但谁也记不得这些树是什么人栽的了。

父亲大力整顿这个市镇，母亲却在振兴家 业，制作美妙的糖公鸡和糖鱼，把它们插在巴里萨木棍儿上，每天两次拿到街上去卖，这时，奥雷连诺却在荒弃的试验室里度过漫长的时刻，孜孜不倦地掌握首饰技 术。他已经长得挺高，哥哥留下的衣服很快不合他的身材了，他就改穿父亲的衣服，诚然，维希塔香不得不替他把衬衫和裤子改窄一些，因为奥雷连诺比父亲和哥哥 都瘦。

进入少年时期，他的嗓音粗了，他也变得沉默寡言、异常孤僻，但是他的眼睛又经常露出紧张的神色，这种神色在他出生的那一天是使他 母亲吃了一惊的。奥雷连诺聚精会神地从事首饰工作，除了吃饭，几乎不到试验室外面去。霍·阿·布恩蒂亚对他的孤僻感到不安，就把房门的钥匙和一点儿钱给了 他，以为儿子可能需要出去找找女人。奥雷连诺却拿钱买了盐酸，制成了王水，给钥匙镀了金。可是，奥雷连诺的古怪比不上阿卡蒂奥和阿玛兰塔的古怪。——这两 个小家伙的乳齿开始脱落，仍然成天跟在印第安人脚边，揪住他们的衣服下摆，硬要说古阿吉洛语，不说西班牙语。“你怨不了别人，”乌苏娜向大夫说。“孩子的 狂劲儿是父母遗传的，”他认为后代的怪诞习惯一点也不比猪尾巴好，就开始抱怨自己倒霉的命运，可是有一次奥色连诺突然拿眼睛盯着她，把她弄得手足无措起 来。

“有人就要来咱们这儿啦，”他说。

象往常一样，儿子预言什么事情，她就用家庭主妇的逻辑破除他的预言。有人到这儿来，那没有什么特别嘛。每天都有几十个外地人经过马孔多，可这并没有叫人操心，他们来到这儿，并不需要预言。然而，奥雷连诺不顾一切逻辑，相信自己的预言。

“我不知道来的人是谁，”他坚持说，“可这个人已在路上啦。”

的确，星期天来了个雷贝卡。她顶多只有十一岁，是跟一些皮货商从马诺尔村来的，经历了艰苦的旅程，这些皮货商受托将这个姑娘连同一封信送到霍·阿·布恩 蒂亚家里，但要求他们帮忙的人究竟是推，他们就说不清楚了。这姑娘的全部行李是一只小衣箱、一把画着鲜艳花朵的木制小摇椅以及一个帆布袋；袋子里老是发出 “咔嚓、咔嚓、咔嚓”的响声——那儿装的是她父母的骸骨。捎绘霍·间·布恩蒂亚的信是某人用特别亲切的口吻写成的，这人说，尽管时间过久，距离颇远，他还 是热爱霍·阿·布恩蒂亚的，觉得自己应当根据基本的人道精神做这件善事——把孤苦伶何的小姑娘送到霍·阿·布恩蒂亚这儿来；这小姑娘是乌苏娜的表侄女，也 就是霍·阿·布恩蒂亚的亲戚，虽是远房的亲戚；因为她是他难忘的朋友尼康诺尔·乌洛阿和他可敬的妻子雷贝卡·蒙蒂埃尔的亲女儿，他们已去天国，现由这小姑 娘把他们的骸骨带去，希望能照基督教的礼仪把它们埋掉。以上两个名字和信未的签名都写得十分清楚，可是霍·阿·布恩蒂亚和乌苏娜都记不得这样的亲戚，也记 不起人遥远的马诺尔村捎信来的这个熟人了。从小姑娘身上了解更多的情况是完全不可能的。她一走进屋子，马上坐在自己的摇椅里，开始咂吮指头，两只惊骇的大 眼睛望着大家，根本不明白人家问她什么。她穿着染成黑色的斜纹布旧衣服和裂开的漆皮鞋。扎在耳朵后面的两络头发，是用黑蝴蝶系住的。脖子上挂着一只香袋， 香袋上有一个汗水弄污的圣像，而右腕上是个铜链条，链条上有一个猛兽的獠牙——防止毒眼的小玩意。她那有点发绿的皮肤和胀鼓鼓、紧绷绷的肚子，证明她健康 不佳和经常挨饿，但别人给她拿来吃的，她却一动不动地继续坐着，甚至没有摸一摸放在膝上的盘子。大家已经认为她是个聋哑姑娘，可是印第安人用自己的语言问 她想不想喝水，她马上转动眼珠，仿佛认出了他们，肯定地点了点头。

他们收留了她，因为没有其他办法。他们决定按照信上对她母亲的称呼， 也管她叫雷贝卡，因为奥雷连诺虽然不厌其烦地在她面前提到一切圣徒的名字，但她对任何一个名字都无反应。当时马孔多没有墓地，因为还没死过一个人，装着骸 骨的袋于就藏了起来，等到有了合适的地方再埋葬，所以长时间里，这袋子总是东藏西放，塞在难以发现的地方，可是经常发出“咔嚓、咔嚓、咔嚓”的响声，就象 下蛋的母鸡咯咯直叫。过了很久雷贝卡才跟这家人的生活协调起来。起初她有个习惯：在僻静的屋角里，坐在摇椅上咂吮指头。任何东西都没引起她的注意，不过， 每过半小时响起钟声的时候，她都惊骇地四面张望，仿佛想在空中发现这种声音似的。

好多天都无法叫她吃饭。谁也不明白她为什么没有饿死， 直到熟悉一切的印第安人发现（因为他们在屋子里用无声的脚步不断地来回走动）雷贝卡喜欢吃的只是院子里的泥土和她用指甲从墙上刨下的一块块石灰。显然，由 于这个恶劣的习惯，父母或者养育她的人惩罚过她，泥上和石灰她都是偷吃的，她知道不对，而且尽量留存一些，无人在旁时可以自由自在地饱餐一顿。从此，他们 对雷贝卡进行了严密的监视，给院子里的泥土浇上牛胆，给房屋的墙壁抹上辛辣的印第安胡椒，恕用这种办法革除姑娘的恶习，但她为了弄到这类吃的，表现了那样 的机智和发明才干，使得乌苏娜不得不采取最有效的措施。她把盛着橙子汁和大黄的锅子整夜放在露天里，次日早饭之前拿这种草药给雷贝卡喝。虽然谁也不会建议 乌苏娜拿这种混合药剂来治疗不良的泥土嗜好，她还是认为任何苦涩的液体进了空肚子，都会在肝脏里引起反应。雷贝卡尽管样子瘦弱，却十分倔强：要她吃药，就 得把她象小牛一样缚住，因为她拼命挣扎，乱抓、乱咬、乱哗，大声叫嚷，今人莫名其妙，据印第安人说，她在骂人，这是古阿吉洛语中最粗鲁的骂人活。乌苏娜知 道了这一点，就用鞭挞加强治疗。所以从来无法断定，究竟什么取得了成效——大黄呢，鞭子呢，或者二者一起；大家知道的只有一点，过了几个星期，雷贝卡开始 出现康复的征象。现在，她跟阿卡蒂奥和阿玛兰塔一块儿玩耍了，她们拿她当做姐姐；她吃饭有味了，会用刀叉了。随后发现，她说西班牙语象印第安语一样流利， 她很能做针线活，还会用自编的可爱歌词照自鸣钟的华尔兹舞曲歌唱。很快，她就似乎成了一个新的家庭成员，她比亲生子女对乌苏娜还亲热；她把阿玛兰塔叫做妹 妹，把阿卡蒂奥叫做弟弟，把奥雷连诺称做叔叔，把霍·阿，布恩蒂亚称做伯伯。这么一来，她和其他的人一样就有权叫做雷贝卡·布恩蒂亚了，——这是她唯一的 名字，至死都体面地叫这个名字。

雷贝卡摆脱了恶劣的泥土嗜好，移居阿玛兰塔和阿卡蒂奥的房间之后，有一天夜里，跟孩子们在一起的印第安 女人偶然醒来，听到犄角里断续地发出一种古怪的声音。她吃惊地从床上一跃而起，担心什么牲畜钻进了屋子，接着便看见雷贝卡坐在摇椅里，把一个指头塞在嘴 里；在黑暗中，她的两只眼睛象猫的眼睛一样闪亮。

维希塔香吓得发呆，在姑娘的眼睛里，她发现了某种疾病的征状，这种疾病的威胁曾使她和弟弟永远离开了那个古老的王国，他俩还是那儿的王位继承人咧。这儿也出现了失眠症。

还没等到天亮，印第安人卡塔乌尔就离开了马孔多。他的姐姐却留了下来，因为宿命论的想法暗示她，致命的疾病反正会跟着她的，不管她逃到多远的地方。然 而，谁也不了解维希塔香的不安。“咱们永远不可睡觉吗？那就更好啦，”霍·阿·布恩蒂亚满意他说。“咱们可从生活中得到更多的东西。”可是印第安女人说 明：患了这种失眠症，最可怕的不是睡不着觉，因为身体不会感到疲乏；最糟糕的是失眠症必然演变成健忘症。她的意思是说，病人经常处于失眠状态，开头会忘掉 童年时代的事儿，然后会忘记东西的名称和用途，最后再也认不得别人，甚至意识不到自己的存在，失去了跟往日的一切联系，陷入一种白痴似的状态。霍·阿·布 恩蒂亚哈哈大笑，差点儿没有笑死，他得出结论说，迷信的印第安人捏造了无数的疾病，这就是其中的一种。可是为了预防万一，谨慎的乌苏娜就让雷贝卡跟其他的 孩子隔离了。

过了几个星期，维希塔香的恐惧过去之后，霍·阿·布恩蒂亚夜间突然发现自己在床上翻来复去合不上眼。乌苏娜也没睡着，问他 是怎么回事，他回答说：“我又在想普鲁登希奥啦。”他俩一分钟也没睡着，可是早上起来却是精神饱满的，立即忘了恶劣的夜晚。吃早饭时，奥雷连诺惊异地说， 他虽在试验室星呆了整整一夜，可是感到自己精神挺好，——他是在试验室里给一枚胸针镀金，打算把它当做生日礼物送给乌苏娜。然而，谁也没有重视这些怪事， 直到两天以后，大家仍在床上合不了眼，才知道自己已经五十多个小时没有睡觉了。

“孩子们也没睡着。这种疫病既然进了这座房子，谁也逃避不了啦，”印第安女人仍用宿命论的口吻说。

的确，全家的人都息了失眠症，乌苏娜曾从母亲那儿得到一些草药知识，就用乌头熬成汤剂，给全家的人喝了，可是大家仍然不能成眠，而且白天站着也做梦。

处在这种半睡半醒的古怪状态中，他们不仅看到自己梦中的形象，而且看到别人梦中的形象。仿佛整座房子都挤满了客人。雷贝卡坐在厨房犄角里的摇椅上，梦见 一个很象她的人，这人穿着白色亚麻布衣服，衬衫领子上有一颗金色钮扣，献给她一柬玫瑰花。他的身边站着一个双手细嫩的女人，她拿出一朵玫瑰花来，佩戴在雷 贝卡的头发上，乌苏娜明白，这男人和女人是姑娘的父母，可是不管怎样竭力辨认，也不认识他们，终于相信以前是从来没有见过他们的。同时，由于注意不够（这 是霍·阿·布恩蒂亚不能原谅自己的），家里制作的糖动物照旧拿到镇上去卖。大人和孩子都快活地吮着有味的绿色公鸡、漂亮的粉红色小鱼、最甜的黄色马儿。这 些糖动物似乎也是患了失眠症的。星期一天亮以后，全城的人已经不睡觉了。起初，谁也不担心。许多的人甚至高兴，——因为当时马孔多百业待兴，时间不够。人 们那么勤奋地工作，在短时间内就把一切都做完了，现在早晨三点就双臂交叉地坐着，计算自鸣钟的华尔兹舞曲有多少段曲调。想睡的人——井非由于疲乏，而是渴 望做梦——采取各种办法把自己弄得精疲力尽，他们聚在一起，不住地絮絮叨叨，一连几小时把同样的奇闻说了又说，大讲特讲白色阉鸡的故事。一直把故事搞得复 杂到了极点。这是一种没完没了的玩耍——讲故事的人问其余的人，他们想不想听白色阉鸡的故事，如果他们回答他“是的”，他就说他要求回答的不是“是的”， 而是要求回答：他们想不想听白色阉鸡的故事；如果他们回答说“不”，他就说他要求回答的不是“不”，而是要求回答：他们想不想听白色阉鸡的故事；如果大家 沉默不语，他就说他要求的不是沉默不语，而是要求回答：他们想不想听白色阉鸡的故事，而且谁也不能走开，因为他说他没有要求他们走开，而是要求回答：他们 想不想听白色阉鸡的故事。就这样，一圈一圈的人，整夜整夜说个没完。

霍·阿·布恩蒂亚知道传染病遍及整个市镇，就把家长们召集起来，告 诉他们有关这种失眠症的常识，并且设法防止这种疾病向邻近的城乡蔓延。于是，大家从一只只山羊身上取下了铃铛——用鹦鹉向阿拉伯人换来的铃铛，把它们挂在 马孔多人口的地方，供给那些不听岗哨劝阻、硬要进镇的人使用。凡是这时经过马孔多街道的外来人都得摇摇铃铛，让失眠症患者知道来人是健康的。他们在镇上停 留的时候，不准吃喝，因为毫无疑问，病从口人嘛，而马孔多的一切食物和饮料都染上了失眠症，采取这些办法，他们就把这种传染病限制在市镇范围之内了。隔离 是严格遵守的，大家逐渐习惯了紧急状态。生活重新上了轨道，工作照常进行，谁也不再担心失去了无益的睡眠习惯。

在几个月中帮助大家跟隐 忘症进行斗争的办法，是奥雷连诺发明的。他发现这种办法也很偶然。奥雷连诺是个富有经验的病人——因为他是失眠症的第一批患者之一——完全掌握了首饰技 术。有一次，他需要一个平常用来捶平金属的小铁砧，可是记不起它叫什么了。父亲提醒他：“铁砧。”奥雷连诺就把这个名字记在小纸片上，贴在铁砧底儿上。现 在，他相信再也不会忘记这个名字了。可他没有想到，这件事儿只是健忘症的第一个表现。过了几天他已觉得，他费了大劲才记起试验室内几乎所有东西的名称。于 是，他给每样东西都贴上标签，现在只要一看签条上的字儿，就能确定这是什么东西了。不安的父亲叫苦连天，说他忘了童年时代甚至印象最深的事儿，奥雷连诺就 把自己的办法告诉他，于是霍·阿·布恩蒂亚首先在自己家里加以采用，然府在全镇推广。他用小刷子蘸了墨水，给房里的每件东西都写上名称：“桌”、“钟”、 “们”、“墙”、“床”、“锅”。然后到畜栏和田地里去，也给牲畜、家禽和植物标上名字：“牛”、“山羊”、“猪”、“鸡”、“木薯”、“香蕉”。人们研 究各种健忘的事物时逐渐明白，他们即使根据签条记起了东西的名称，有朝一日也会想不起它的用途。随后，他们就把签条搞得很复杂了。一头乳牛脖子上挂的牌 子，清楚他说明马孔多居民是如何跟健忘症作斗争的：“这是一头乳牛。每天早晨挤奶，就可得到牛奶，把牛奶煮沸，掺上咖啡，就可得牛奶咖啡。”就这样，他们 生活在经常滑过的现实中，借助字儿能把现实暂时抓住，可是一旦忘了字儿的意义，现实也就难免忘诸脑后了。

市镇入口的地方挂了一块脾子： “马孔多”，中心大街上挂了另一块较大的牌子：“”上帝存在“。所有的房屋都画上了各种符号，让人记起各种东西。然而，这一套办法需要密切的注意力，还要 耗费很在的精神，所以许多人就陷入自己的幻想世界，——这对他们是不太实际的，却是更有安慰的。推广这种自欺的办法，最起劲的是皮拉·苔列娜，她想出一种 用纸牌测知过去的把戏，就象她以前用纸牌预卜未来一样。由于她那些巧妙的谎言，失眠的马孔多居民就处于纸牌推测的世界，这些推测含糊不清，互相矛盾，面在 这个世界中，只能模糊地想起你的父亲是个黑发男人，是四月初来到这儿的；母亲是个黝黑的女人，左手戴着一枚金戒指，你出生的日期是某月的最后一个星期二， 那一天百灵鸟在月桂树上歌唱。霍·阿·布恩蒂亚被这种安慰的办法击败了，他为了对抗，决定造出一种记忆机器，此种机器是他以前打算制造出来记住吉卜赛人的 一切奇异发明的，机器的作用原理就是每天重复在生活中获得的全部知识。霍·阿·布恩蒂亚把这种机械设想成一本旋转的字典，人呆在旋转轴上，利用把手操纵字 典，——这样，生活所需的一切知识短时间内就在眼前经过，他已写好了几乎一万四千张条目卡，这时，从沼泽地带伸来的路上，出现一个样子古怪的老人儿，摇着 悲哀的铃铛，拎着一只绳子系住的、胀鼓鼓的箱子，拉着一辆用黑布遮住的小车子。他径直朝霍·阿·布恩蒂亚的房子走来。

维希塔香给老头儿 开了门，却不认得他，把他当成一个商人，老头儿还没听说这个市镇绝望地陷进了健忘症的漩涡，不知道在这儿是卖不出什么东西的。这是一个老朽的人。尽管他的 嗓音犹豫地发颤，双乎摸摸索索的，但他显然是从另一个世界来的，那里的人既能睡觉，又能记忆。霍·阿·布恩蒂亚出来接见老头儿的时候，老头儿正坐在客厅 里，拿破旧的黑帽子扇着，露出同情的样儿，注意地念了念贴在墙上的字条。霍·阿·布恩蒂亚非常恭敬地接待他，担心自己从前认识这个人，现在却把他给忘了。 然而客人识破了他的佯装，感到自己被他忘却了，——他知道这不是心中暂时的忘却，而是另一种更加冷酷的、彻底的忘却，也就是死的忘却。

接着，他一切都明白了。他打开那只塞满了不知什么东西的箱子，从中掏出一个放着许多小瓶子的小盒子。他把一小瓶颜色可爱的药水递给房主人，房主人把它喝 了，马上恍然大悟。霍·阿·布恩蒂亚两眼噙满悲哀的泪水，然后才看出自己是在荒谬可笑的房间里，这儿的一切东西都贴上了字条；他羞愧地看了看墙上一本正经 的蠢话，最后才兴高采烈地认出客人就是梅尔加德斯。

马孔多庆祝记忆复原的时候，霍·阿·布恩蒂亚和梅尔加德斯恢复了往日的友谊。吉卜赛 人打算留居镇上。他的确经历过死亡，但是忍受不了孤独，所以回到这儿来了。因为他忠于现实生活，失去了自己的神奇本领，被他的部族抛弃，他就决定在死神还 没发现的这个角落里得到一个宁静的栖身之所，把自己献给银版照相术。霍·阿·布恩蒂亚根本没有听说过这样的发明。可是，当他看见自己和全家的人永远印在彩 虹色的金属版上时，他惊得说不出话了；霍·阿·布恩蒂亚有一张锈了的照相底版就是这时的——蓬乱的灰色头发，铜妞扣扣上的浆领衬衫，一本正经的惊异表情。 乌苏娜笑得要死，认为他象“吓破了胆的将军。”说真的，在那晴朗的十二月的早晨，梅尔加德斯拍照的时候，霍·阿·布恩蒂亚确实吓坏了：他生怕人像移到金属 版上，人就会逐渐消瘦。不管多么反常，乌苏娜这一次却为科学辩护，竭力打消丈夫脑瓜里的荒谬想法。他忘了一切旧怨，决定让梅尔加德斯住在他们家里。然而， 乌苏娜自己从不让人给她拍照，因为（据她自己的说法）她不愿留下像来成为子孙的笑柄。那天早晨，她给孩子们穿上好衣服，在他们脸上搽了粉，让每人喝了一匙 骨髓汤，使他们能在梅尔加德斯奇异的照相机前面凝然不动地站立几乎两分钟。在这张“全家福”（这是过去留下的唯一的照片）上，奥雷连诺穿着黑色丝绒衣服， 站在阿玛兰塔和雷贝卡之间，他的神情倦怠，目光明澈，多年以后，他就是这副神态站在行刑队面前的。可是，照片上的青年当时还没听到命运的召唤，他只是一个 能干的首饰匠，由于工作认真，在整个沼泽地带都受到尊重。他的作坊同时是梅尔加德斯的试验室，这儿几乎听不到他的声音。在瓶子的当嘟声和盘子的敲击声中， 在接连不断的灾难中：酸溢出来了，溴化银浪费掉了，当他的父亲和吉卜赛人大声争论纳斯特拉达马斯的预言时，奥雷连诺似乎呆在另一个世界里。奥雷连诺忘我地 工作，善于维护自己的利益，因此在短时期内，他挣的钱就超过了乌苏娜出售糖动物的收益。大家觉得奇怪的只有一点——他已经是个完全成熟的人，为什么至今不 结交女人，的确，他还没有女人。

过了几个月，那个弗兰西斯科人又来到了马孔多；他是个老流浪汉，差不多两百岁了。他常常路过马孔多，带 来自编的歌曲。在这些歌曲中，弗兰西斯科人非常详细地描绘了一些事情，这些事情都发生在他途中经过的地方——从马诺尔村到沼泽地另一边的城乡里，所以，谁 想把信息传给熟人，或者想把什么家事公诸于世，只消付两分钱，弗兰西斯科人就可把它列入自己的节目。有一天傍晚，乌苏娜听唱时希望知道儿子的消息，却完全 意外地听到了自己母亲的死讯。“弗兰西斯科人”

这个绰号的由来，是他在编歌比赛中战胜过魔鬼，他的真名实姓是谁也不知道的；

失眠症流行时，他就从马孔多消失了，现在又突然来到了卡塔林诺游艺场。大家都去听他吟唱，了解世界上发生的事儿。跟弗兰西斯科人一起来到马孔多的，有一 个妇人和一个年轻的混血姑娘；妇人挺胖，是四个印第安人用摇椅把她抬来的；她头上撑着一把小伞，遮住阳光。混血姑娘却是一副可怜相。这一次，奥雷连诺也来 到了卡塔林诺游艺场。弗兰西斯科人端坐在一群听众中间，仿佛一条硕大的变色龙。

他用老年人颤抖的声调歌唱，拿华特·赖利在圭亚那给他的 那个古老的手风琴伴奏，用步行者的大脚掌打着拍子；他的脚掌已给海盐弄得裂开了。屋子深处看得见另一个房间的门，一个个男人不时挨次进去，摇椅抬来的那个 胖妇人坐在门口，默不作声地扇着扇子，卡塔林诺耳后别着一朵假玫瑰，正在卖甘蔗酒，并且利用一切借口走到男人跟前，把手伸到他们身上去摸不该摸的地方。时 到午夜，热得难受。奥雷连诺听完一切消息，可是没有发现任何跟自己的家庭有关的事。他已经准备离开，这时那个妇人却用手招呼他。

“你也进去吧，”她说。“只花两角钱。”

奥雷连诺把钱扔到胖妇人膝上的一只匣子里，打开了房门，自己也不知道去干什么。床上躺着那个年轻的混血姑娘，浑身赤裸，她的胸脯活象母狗的乳头。在奥雷 连诺之前，这儿已经来过六十三个男人，空气中充满了那么多的碳酸气，充满了汗水和叹息的气味，已经变得十分污浊；姑娘取下湿透了的床单，要求奥雷连诺抓住 床唯的一头。床单挺重，好象湿帆布。他们抓住床单的两头拧了又拧，它才恢复了正常的重量。然后，他们翻过垫子，汗水却从另一面流了出来。奥雷连诺巴不得把 这一切没完没了地干下去。爱情的奥秘他从理论上是知道的，但是他的膝头却在战粟，他勉强才能姑稳脚跟。姑娘拾掇好了床铺，要他脱掉衣服时，他却给她作了混 乱的解释：“是他们要我进来的。他们要我把两角钱扔在匣子里，叫我不要耽搁。”姑娘理解他的混乱状态，低声说道：“你出去的时候，再扔两角钱，就可呆得久 一点儿。”奥雷连诺羞涩难堪地脱掉了衣服；他总是以为向己的裸体比不上哥哥的裸体。虽然姑娘尽心竭力，他却感到肉己越来越冷漠和孤独。“我再扔两角钱 吧，”他完全绝望地咕噜着说。姑娘默不作声地向他表示感谢。她皮包骨头，脊背磨出了血。由于过度疲劳，呼吸沉重、断断续续。两年前，在离马孔多很远的地 方，有一天晚上她没熄灭蜡烛就睡着了，醒来的时候，周围一片火焰，她和一个把她养大的老大娘一起居住的房子，烧得精光。从此以后，老大娘就把她带到一个个 城镇，让她跟男人睡一次觉捞取两角钱，用来弥补房屋的损失。按照姑娘的计算，她还得再这样生活十年左右，一夜接待七十个男人，因为除了偿债，还得支付她俩 的路费和膳食费以及印第安人的抬送费。老大娘第二次敲门的时候，奥雷连诺什么也没做就走出房间，好不容易忍住了泪水，这天夜里，他睡不着觉，老是想着混血 姑娘，同时感到怜悯和需要。他渴望爱她和保护她。他被失眠和狂热弄得疲惫不堪，次日早晨就决定跟她结婚，以便把她从老大娘的控制下解救出来，白个儿每夜都 得到她给七十个男人的快乐。可是早上十点他来到卡塔林诺游艺场的时候，姑娘已经离开了马孔多。

时间逐渐冷却了他那热情的、轻率的打算， 但是加强了他那希望落空的痛苦感觉。他在工作中寻求解脱。为了掩饰自己不中用的耻辱，他顺人了一辈子打光棍的命运。这时，梅尔加德斯把马孔多一切值得拍照 的都拍了照，就将银版照相器材留给霍·阿·布恩蒂亚进行荒唐的试验：后者决定利用银版照相术得到上帝存在的科学证明。他相信，拿屋内不同地方拍的照片进行 复杂的加工，如果上帝存在的话，他迟早准会得到上帝的照片，否则就永远结束有关上帝存在的一切臆想。梅尔加德斯却在深入研究纳斯特拉达马斯的理论。他经常 坐到很晚，穿着褪了色的丝绒坎肩直喘粗气，用他干瘦的鸟爪在纸上潦草地写着什么；他手上的戒指已经失去往日的光彩。有一天夜晚，他觉得他偶然得到了有关马 孔多未来的启示。马孔多将会变成一座辉煌的城市，有许多高大的玻璃房子，城内甚至不会留下布恩蒂亚家的痕迹。

“胡说八道，”霍·阿·布恩蒂亚气恼他说。“不是玻璃房子，而是我梦见的那种冰砖房子，并且这儿永远都会有布思蒂亚家的人，Peromniaseculasecul-orumo！”（拉丁语：永远永远）乌苏娜拼命想给这个怪人的住所灌输健全的思想。

她添了一个大炉灶，除了生产糖动物，开始烤山整篮整篮的面包和大堆大堆各式各样的布丁、奶油蛋白松饼和饼干——这一切在几小时内就在通往沼泽地的路上卖 光了。尽管乌苏娜已经到了应当休息的年岁，但她年复一年变得越来越勤劳了，全神贯注在兴旺的生意上，有一天傍晚，印第安女人正帮她把糖掺在生面里，她漫不 经心地望着窗外，突然看见院子里有两个似乎陌生的姑娘，都很年轻、漂亮，正在落日的余晖中绣花。这是雷贝卡和阿玛兰塔。她们刚刚脱掉穿了三年的悼念外祖母 的孝服。花衣服完全改变了她们的外貌。出乎一切预料，雷贝卡在姿色上超过了阿玛兰塔，她长着宁静的大眼睛、光洁的皮肤和具有魔力的手：她的手仿佛用看不见 的丝线在绣架的布底上刺绣。较小的阿玛兰塔不够雅致，但她从已故的外祖母身上继承了天生的高贵和自尊心。呆在她们旁边的是阿卡蒂奥，他身上虽已显露了父亲 的体魄，但看上去还是个孩子。他在奥雷连诺的指导下学习首饰技术，奥雷连诺还教他读书写字。乌苏娜明白，她家里满是成年的人，她的孩子们很快就要结婚，也 要养孩子，全家就得分开，因为这座房子不够大家住了。于是，她拿出长年累月艰苦劳动积攒的钱，跟工匠们商量好，开始扩充住宅。她吩咐增建：一间正式客厅 ——用来接待客人：另一间更舒适、凉爽的大厅——供全家之用，一个饭厅，拥有一张能坐十二人的桌子；九间卧室，窗户都面向庭院；一道长廊，由玫瑰花圃和宽 大的栏杆（栏杆上放着一盆盆碳类植物和秋海棠）挡住晌午的阳光。而且，她还决定扩大厨房，安置两个炉灶；拆掉原来的库房（皮拉·苔列娜曾在里面向霍·阿卡 蒂奥预言过他的未来），另盖一间大一倍的库房，以便家中经常都有充足的粮食储备。在院子里，在大栗树的浓荫下面，乌苏娜嘱咐搭两个浴棚：一个女浴棚，一个 男浴棚，而星后却是宽敞的马厩、铁丝网围住的鸡窝和挤奶棚，此外有个四面敞开的鸟笼，偶然飞来的鸟儿高兴栖息在那儿就栖息在那儿。乌苏娜带领着几十名泥瓦 匠和木匠，仿佛染上了大大的“幻想热”，决定光线和空气进人屋子的方位，划分面帆完全不受限。马孔多建村时修盖的这座简陋房子，堆满了各种工具和建筑材 料，工人们累得汗流浃背，老是提醒旁人不要妨碍他们干活，而他们总是碰到那只装着骸骨的袋子，它那沉闷的咔嚓声简直叫人恼火。谁也不明白，在这一片混乱 中，在生石灰和沥青的气味中，地下怎会立起一座房子，这房子不仅是全镇最大的，而且是沼泽地区最凉爽宜人的。最不理解这一点的是霍·阿·布恩蒂亚，甚至在 大变动的高潮中，他也没有放弃突然摄到上帝影像的尝试。新房子快要竣工的时候，乌苏娜把他拉出了幻想的世界，告诉他说，她接到一道命令：房屋正面必须刷成 蓝色，不能刷成他们希望的白色。她把正式公文给他看。霍·阿·布恩蒂亚没有马上明白他的妻子说些什么，首先看了看纸儿上的签字。

“这个人是谁？”他问。

“镇长，”乌苏娜怏怏不乐地回答。“听说他是政府派来的官儿。”

阿·摩斯柯特镇长先生是不声不响地来到马孔多的。第一批阿拉伯人来到这儿，用小玩意儿交换鹦鹉的时候，有个阿拉伯人开了一家雅各旅店，阿·摩斯柯特首先 住在这个旅店里，第二天才租了一个门朝街的小房间，离布恩蒂亚的房子有两个街区。他在室内摆上从雅各旅店买来的桌子和椅子，把带来的共和国国徽钉在墙上， 并且在门上刷了“镇长”二字。他的第一道命令就是要所有的房屋刷成蓝色，借以庆祝国家独立的周年纪念。

霍·阿·布恩蒂亚拿着复写的命令来找镇长，正碰见他在小办公室的吊床上睡午觉。“这张纸儿是你写的吗？”霍·阿·布恩蒂亚问。阿·摩斯柯特是个上了岁数的人，面色红润，显得胆怯，作了肯定的问答。“凭什么权力？”霍·阿·布恩蒂亚又问。

阿·摩斯柯特从办公桌抽屉内拿出一张纸来，递给他看。“兹派该员前往上述市镇执行镇长职务。”霍·阿·布恩蒂亚对这委任状看都不看一眼。

“在这个市镇上，我们不靠纸儿发号施令，”他平静地回答。“请你永远记住：我们不需要别人指手画脚，我们这儿的事用不着别人来管。”

阿·摩斯柯特先生保持镇定，霍·阿·布恩蒂亚仍然没有提高声音，向他详细他讲了讲：他们如何建村，如何划分土地、开辟道路，做了应做的一切，从来没有麻 烦过任何政府。谁也没有来麻烦过他们。“我们是爱好和平的人，我们这儿甚至还没死过人咧。”霍·阿·布恩蒂亚说。“你能看出，马孔多至今没有墓地。”他没 有抱怨政府，恰恰相反，他高兴没有人来妨碍他们安宁地发展，希望今后也是如此，因为他们建立马孔多村，不是为了让别人来告诉他们应该怎么办的。阿，摩斯柯 特先生穿上象裤子一样白的祖布短上衣，一分钟也没忘记文雅的举止。

“所以，如果你想留在这个镇上做一个普通的居民，我们完全欢迎。”霍·阿·布恩蒂亚最后说。“可是，如果你来制造混乱，强迫大伙儿把房子刷成蓝色，那你就拿起自己的行李，回到你来的地方去，我的房子将会白得象一只鸽子。”

阿·摩斯柯特先生脸色发白。他倒退一步，咬紧牙关，有点激动他说：

“我得警告你，我有武器。”

霍·阿·布恩蒂亚甚至没有发觉，他的双手刹那问又有了年轻人的力气，从前他靠这种力气曾把牲口按倒在地，他一把揪住阿·摩斯柯特的衣领，把他举到自己眼前。

“我这么做，”他说，“因为我认为我已到了余年，与其拖一个死人，不如花几分钟拖一个活人。”

就这样，他把悬在衣领上的阿·摩斯柯特先生沿着街道中间拎了过去，在马孔多到沼泽地的路上他才让他双脚着地。过了一个星期，阿·摩斯柯特又来了，带着六 名褴褛、赤足、持枪的士兵，还有一辆牛车，车上坐着他的妻子和七个女儿。随后又来了两辆牛车，载着家具、箱子他和其他家庭用具。镇长暂时把一家人安顿在雅 各旅店里，随后找到了房子，才在门外安了两名卫兵，开始办公，马孔多的老居民决定撵走这些不速之客，就带着自己年岁较大的几子去找霍·阿·布恩蒂亚，希望 他担任指挥。可是霍·阿·布恩蒂亚反对他们的打算，因为据他解释，阿·摩斯柯特先生既然跟妻子和女儿一起回来了，在他的一家人面前侮辱他，就不是男子汉大 丈夫了。事情应当和平解决。

奥雷连诺自愿陪伴父亲。这时，他已长了尖端翘起的黑胡髭，嗓音洪亮，这种嗓音在战争中是会使他大显威风的。 他们没带武器，也没理睬卫兵，径直跨进了镇长办公室，阿·摩斯柯特先生毫不慌乱。他把他们介绍给他的两个女儿；她们是偶然来到办公室的：一个是十六岁的安 芭萝，象她母亲一样满头乌发，一个是刚满九岁的雷麦黛丝，这小姑娘挺可爱，皮肤细嫩，两眼发绿。姐妹俩都挺文雅，很讲礼貌。布恩蒂亚父子两人刚刚进来，她 俩还没听到介绍，就给客人端来椅子。可是他们不愿坐下。

“好啦，朋友，”霍·阿·布恩蒂亚说，“我们让你住在这儿，但这并不是因为门外站着几个带枪的强盗，而是由于尊敬你的夫人和女儿。”

阿·摩斯柯特张口结舌，可是霍·阿·布恩蒂亚没有让他反驳。

“但是我们必须向你提出两个条件，”他补充说。“第一：每个人想把自己的房子刷成什么颜色就是什么颜色。第二：大兵们立即离开马孔多，镇上的秩序由我们负责。”

镇长起誓似的举起手来。

“这是真话？”

“敌人的话，”霍·阿·布恩蒂亚说。接着又苦楚地添了一句：“因为我得告诉你一点：你和我还是敌人。”

就在这一天下午，士兵们离开了市镇。过了几天，霍·阿·布恩蒂亚为镇长一家人找到了一座房子。除了奥雷连诺。大家都平静下来。镇长的小女儿雷麦黛丝，就 年龄来说，也适于做奥雷连诺的女儿，可是她的形象却留在他的心里，使他经常感到痛苦。这是肉体上的感觉，几乎妨碍他走路，仿佛一块石子掉进了他的鞋里。
"""

ENGLISH_PROMPT="""
461 U.S. 238 (1983) OLIM ET AL. v. WAKINEKONA No. 81-1581. Supreme Court of United States. Argued January 19, 1983. Decided April 26, 1983. CERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT *239 Michael A. Lilly, First Deputy Attorney General of Hawaii, argued the cause for petitioners. With him on the brief was James H. Dannenberg, Deputy Attorney General. Robert Gilbert Johnston argued the cause for respondent. With him on the brief was Clayton C. Ikei.[*] *240 JUSTICE BLACKMUN delivered the opinion of the Court. The issue in this case is whether the transfer of a prisoner from a state prison in Hawaii to one in California implicates a liberty interest within the meaning of the Due Process Clause of the Fourteenth Amendment. I A Respondent Delbert Kaahanui Wakinekona is serving a sentence of life imprisonment without the possibility of parole as a result of his murder conviction in a Hawaii state court. He also is serving sentences for various other crimes, including rape, robbery, and escape. At the Hawaii State Prison outside Honolulu, respondent was classified as a maximum security risk and placed in the maximum control unit. Petitioner Antone Olim is the Administrator of the Hawaii State Prison. The other petitioners constituted a prison 'Program Committee.' On August 2, 1976, the Committee held hearings to determine the reasons for a breakdown in discipline and the failure of certain programs within the prison's maximum control unit. Inmates of the unit appeared at these hearings. The Committee singled out respondent and another inmate as troublemakers. On August 5, respondent received notice that the Committee, at a hearing to be held on August 10, would review his correctional program to determine whether his classification within the system should be changed and whether he should be transferred to another Hawaii facility or to a mainland institution. *241 The August 10 hearing was conducted by the same persons who had presided over the hearings on August 2. Respondent retained counsel to represent him. The Committee recommended that respondent's classification as a maximum security risk be continued and that he be transferred to a prison on the mainland. He received the following explanation from the Committee: 'The Program Committee, having reviewed your entire file, your testimony and arguments by your counsel, concluded that your control classification remains at Maximum. You are still considered a security risk in view of your escapes and subsequent convictions for serious felonies. The Committee noted the progress you made in vocational training and your expressed desire to continue in this endeavor. However your relationship with staff, who reported that you threaten and intimidate them, raises grave concerns regarding your potential for further disruptive and violent behavior. Since there is no other Maximum security prison in Hawaii which can offer you the correctional programs you require and you cannot remain at [the maximum control unit] because of impending construction of a new facility, the Program Committee recommends your transfer to an institution on the mainland.' App. 7-8. Petitioner Olim, as Administrator, accepted the Committee's recommendation, and a few days later respondent was transferred to Folsom State Prison in California. B Rule IV of the Supplementary Rules and Regulations of the Corrections Division, Department of Social Services and Housing, State of Hawaii, approved in June 1976, recites that the inmate classification process is not concerned with punishment. Rather, it is intended to promote the best interests *242 of the inmate, the State, and the prison community.[1] Paragraph 3 of Rule IV requires a hearing prior to a prison transfer involving 'a grievous loss to the inmate,' which the Rule defines 'generally' as 'a serious loss to a reasonable man.' App. 21.[2] The Administrator, under ¶ 2 of the Rule, is required to establish 'an impartial Program Committee' to conduct such a hearing, the Committee to be 'composed of at least three members who were not actively involved in the process by which the inmate . . . was brought before the Committee.' App. 20. Under ¶ 3, the Committee must give the inmate written notice of the hearing, permit him, with certain stated exceptions, to confront and cross-examine witnesses, afford him an opportunity to be heard, and apprise him of the Committee's findings. App. 21-24.[3] The Committee is directed to make a recommendation to the Administrator, who then decides what action to take: '[The Administrator] may, as the final decisionmaker: '(a) Affirm or reverse, in whole or in part, the recommendation; or '(b) hold in abeyance any action he believes jeopardizes the safety, security, or welfare of the staff, inmate *243. . . , other inmates . . . , institution, or community and refer the matter back to the Program Committee for further study and recommendation.' Rule IV, ¶ 3d(3), App. 24. The regulations contain no standards governing the Administrator's exercise of his discretion. See Lono v. Ariyoshi, 63 Haw. 138, 144-145, 621 P. 2d 976, 980-981 (1981). C Respondent filed suit under 42 U. S. C. § 1983 against petitioners as the state officials who caused his transfer. He alleged that he had been denied procedural due process because the Committee that recommended his transfer consisted of the same persons who had initiated the hearing, this being in specific violation of Rule IV, ¶ 2, and because the Committee was biased against him. The United States District Court for the District of Hawaii dismissed the complaint, holding that the Hawaii regulations governing prison transfers do not create a substantive liberty interest protected by the Due Process Clause. 459 F. Supp. 473 (1978).[4] The United States Court of Appeals for the Ninth Circuit, by a divided vote, reversed. 664 F. 2d 708 (1981). It held that Hawaii had created a constitutionally protected liberty interest by promulgating Rule IV. In so doing, the court declined to follow cases from other Courts of Appeals holding that certain procedures mandated by prison transfer regulations do not create a liberty interest. See, e. g., Cofone v. Manson, 594 F. 2d 934 (CA2 1979); Lombardo v. Meachum, 548 F. 2d 13 (CA1 1977). The court reasoned that Rule IV gives Hawaii prisoners a justifiable expectation that they will not be transferred to the mainland absent a hearing, before an impartial committee, concerning the facts alleged in the *244 prehearing notice.[5] Because the Court of Appeals' decision created a conflict among the Circuits, and because the case presents the further question whether the Due Process Clause in and of itself protects against interstate prison transfers, we granted certiorari. 456 U. S. 1005 (1982). II In Meachum v. Fano, 427 U. S. 215 (1976), and Montanye v. Haymes, 427 U. S. 236 (1976), this Court held that an intrastate prison transfer does not directly implicate the Due Process Clause of the Fourteenth Amendment. In Meachum, inmates at a Massachusetts medium security prison had been transferred to a maximum security prison in that Commonwealth. In Montanye, a companion case, an inmate had been transferred from one maximum security New York prison to another as punishment for a breach of prison rules. This Court rejected 'the notion that any grievous loss visited upon a person by the State is sufficient to invoke the procedural protections of the Due Process Clause.' Meachum, 427 U. S., at 224 (emphasis in original). It went on to state: 'The initial decision to assign the convict to a particular institution is not subject to audit under the Due Process Clause, although the degree of confinement in one prison may be quite different from that in another. The conviction has sufficiently extinguished the defendant's liberty *245 interest to empower the State to confine him in any of its prisons. 'Neither, in our view, does the Due Process Clause in and of itself protect a duly convicted prisoner against transfer from one institution to another within the state prison system. Confinement in any of the State's institutions is within the normal limits or range of custody which the conviction has authorized the State to impose.' Id., at 224-225 (emphasis in original). The Court observed that, although prisoners retain a residuum of liberty, see Wolff v. McDonnell, 418 U. S. 539, 555-556 (1974), a holding that 'any substantial deprivation imposed by prison authorities triggers the procedural protections of the Due Process Clause would subject to judicial review a wide spectrum of discretionary actions that traditionally have been the business of prison administrators rather than of the federal courts.' 427 U. S., at 225 (emphasis in original). Applying the Meachum and Montanye principles in Vitek v. Jones, 445 U. S. 480 (1980), this Court held that the transfer of an inmate from a prison to a mental hospital did implicate a liberty interest. Placement in the mental hospital was 'not within the range of conditions of confinement to which a prison sentence subjects an individual,' because it brought about 'consequences . . . qualitatively different from the punishment characteristically suffered by a person convicted of crime.' Id., at 493. Respondent argues that the same is true of confinement of a Hawaii prisoner on the mainland, and that Vitek therefore controls. We do not agree. Just as an inmate has no justifiable expectation that he will be incarcerated in any particular prison within a State, he has no justifiable expectation that he will be incarcerated in any particular State.[6] Often, confinement *246 in the inmate's home State will not be possible. A person convicted of a federal crime in a State without a federal correctional facility usually will serve his sentence in another State. Overcrowding and the need to separate particular prisoners may necessitate interstate transfers. For any number of reasons, a State may lack prison facilities capable of providing appropriate correctional programs for all offenders. Statutes and interstate agreements recognize that, from time to time, it is necessary to transfer inmates to prisons in other States. On the federal level, 18 U. S. C. § 5003(a) authorizes the Attorney General to contract with a State for the transfer of a state prisoner to a federal prison, whether in that State or another. See Howe v. Smith, 452 U. S. 473 (1981).[7] Title 18 U. S. C. § 4002 (1976 ed. and Supp. V) permits the Attorney General to contract with any State for the placement of a federal prisoner in state custody for up to three years. Neither statute requires that the prisoner remain in the State in which he was convicted and sentenced. On the state level, many States have statutes providing for the transfer of a state prisoner to a federal prison, e. g., Haw. Rev. Stat. § 353-18 (1976), or another State's prison, e. g., Alaska Stat. Ann. § 33.30.100 (1982). Corrections compacts between States, implemented by statutes, authorize incarceration of a prisoner of one State in another State's prison. See, e. g., Cal. Penal Code Ann. § 11189 (West 1982) (codifying Interstate Corrections Compact); § 11190 (codifying Western Interstate Corrections Compact); Conn. Gen. *247 Stat. § 18-102 (1981) (codifying New England Interstate Corrections Compact); § 18-106 (codifying Interstate Corrections Compact); Haw. Rev. Stat. § 355-1 (1976) (codifying Western Interstate Corrections Compact); Idaho Code § 20-701 (1979) (codifying Interstate Corrections Compact); Ky. Rev. Stat. § 196.610 (1982) (same). And prison regulations such as Hawaii's Rule IV anticipate that inmates sometimes will be transferred to prisons in other States. In short, it is neither unreasonable nor unusual for an inmate to serve practically his entire sentence in a State other than the one in which he was convicted and sentenced, or to be transferred to an out-of-state prison after serving a portion of his sentence in his home State. Confinement in another State, unlike confinement in a mental institution, is 'within the normal limits or range of custody which the conviction has authorized the State to impose.' Meachum, 427 U. S., at 225.[8] Even when, as here, the transfer involves long distances and an ocean crossing, the confinement remains within constitutional limits. The difference between such a transfer and an intrastate or interstate transfer of *248 shorter distance is a matter of degree, not of kind,[9] and Meachum instructs that 'the determining factor is the nature of the interest involved rather than its weight.' 427 U. S., at 224. The reasoning of Meachum and Montanye compels the conclusion that an interstate prison transfer, including one from Hawaii to California, does not deprive an inmate of any liberty interest protected by the Due Process Clause in and of itself. III The Court of Appeals held that Hawaii's prison regulations create a constitutionally protected liberty interest. In Meachum, however, the State had 'conferred no right on the *249 prisoner to remain in the prison to which he was initially assigned, defeasible only upon proof of specific acts of misconduct,' 427 U. S., at 226, and 'ha[d] not represented that transfers [would] occur only on the occurrence of certain events,' id., at 228. Because the State had retained 'discretion to transfer [the prisoner] for whatever reason or for no reason at all,' ibid., the Court found that the State had not created a constitutionally protected liberty interest. Similarly, because the state law at issue in Montanye 'impose[d] no conditions on the discretionary power to transfer,' 427 U. S., at 243, there was no basis for invoking the protections of the Due Process Clause. These cases demonstrate that a State creates a protected liberty interest by placing substantive limitations on official discretion. An inmate must show 'that particularized standards or criteria guide the State's decisionmakers.' Connecticut Board of Pardons v. Dumschat, 452 U. S. 458, 467 (1981) (BRENNAN, J., concurring). If the decisionmaker is not 'required to base its decisions on objective and defined criteria,' but instead 'can deny the requested relief for any constitutionally permissible reason or for no reason at all,' ibid., the State has not created a constitutionally protected liberty interest. See id., at 466-467 (opinion of the Court); see also Vitek v. Jones, 445 U. S., at 488-491 (summarizing cases). Hawaii's prison regulations place no substantive limitations on official discretion and thus create no liberty interest entitled to protection under the Due Process Clause. As Rule IV itself makes clear, and as the Supreme Court of Hawaii has held in Lono v. Ariyoshi, 63 Haw., at 144-145, 621 P. 2d, at 980-981, the prison Administrator's discretion to transfer an inmate is completely unfettered. No standards govern or restrict the Administrator's determination. Because the Administrator is the only decisionmaker under Rule IV, we need not decide whether the introductory paragraph *250 of Rule IV, see n. 1, supra, places any substantive limitations on the purely advisory Program Committee.[10] The Court of Appeals thus erred in attributing significance to the fact that the prison regulations require a particular kind of hearing before the Administrator can exercise his unfettered discretion.[11] As the United States Court of Appeals for the Seventh Circuit recently stated in Shango v. Jurich, 681 F. 2d 1091, 1100-1101 (1982), '[a] liberty interest is of course a substantive interest of an individual; it cannot be the right to demand needless formality.'[12] Process is not an end in itself. Its constitutional purpose is to protect a substantive interest to which the individual has a legitimate claim of entitlement. See generally Simon, Liberty and Property in the Supreme Court: A Defense of Roth and Perry, 71 Calif. L. Rev. 146, 186 (1983). If officials may transfer a prisoner 'for whatever reason or for no reason at all,' Meachum, 427 U. S., at 228, there is no such interest for process to protect. The State may choose to require procedures for reasons other than protection against deprivation of substantive *251 rights, of course,[13] but in making that choice the State does not create an independent substantive right. See Hewitt v. Helms, 459 U. S. 460, 471 (1983). IV In sum, we hold that the transfer of respondent from Hawaii to California did not implicate the Due Process Clause directly, and that Hawaii's prison regulations do not create a protected liberty interest.[14] Accordingly, the judgment of the Court of Appeals is Reversed. JUSTICE MARSHALL, with whom JUSTICE BRENNAN joins, and with whom JUSTICE STEVENS joins as to Part I, dissenting. In my view, the transfer of respondent Delbert Kaahanui Wakinekona from a prison in Hawaii to a prison in California implicated an interest in liberty protected by the Due Process Clause of the Fourteenth Amendment. I respectfully dissent. I An inmate's liberty interest is not limited to whatever a State chooses to bestow upon him. An inmate retains a significant residuum of constitutionally protected liberty following his incarceration independent of any state law. As we stated in Wolff v. McDonnell, 418 U. S. 539, 555-556 (1974): '[A] prisoner is not wholly stripped of constitutional protections when he is imprisoned for crime. There is no iron curtain drawn between the Constitution and the prisons *252 of this country. . . . [Prisoners] may not be deprived of life, liberty, or property without due process of law.' In determining whether a change in the conditions of imprisonment implicates a prisoner's retained liberty interest, the relevant question is whether the change constitutes a sufficiently 'grievous loss' to trigger the protection of due process. Vitek v. Jones, 445 U. S. 480, 488 (1980). See Morrissey v. Brewer, 408 U. S. 471, 481 (1972), citing Joint Anti-Fascist Refugee Committee v. McGrath, 341 U. S. 123, 168 (1951) (Frankfurter, J., concurring). The answer depends in part on a comparison of 'the treatment of the particular prisoner with the customary, habitual treatment of the population of the prison as a whole.' Hewitt v. Helms, 459 U. S. 460, 486 (1983) (STEVENS, J., dissenting). This principle was established in our decision in Vitek, which held that the transfer of an inmate from a prison to a mental hospital implicated a liberty interest because it brought about 'consequences . . . qualitatively different from the punishment characteristically suffered by a person convicted of crime.' 445 U. S., at 493. Because a significant qualitative change in the conditions of confinement is not 'within the range of conditions of confinement to which a prison sentence subjects an individual,' ibid., such a change implicates a prisoner's protected liberty interest. There can be little doubt that the transfer of Wakinekona from a Hawaii prison to a prison in California represents a substantial qualitative change in the conditions of his confinement. In addition to being incarcerated, which is the ordinary consequence of a criminal conviction and sentence, Wakinekona has in effect been banished from his home, a punishment historically considered to be 'among the severest.'[1] For an indeterminate period of time, possibly the *253 rest of his life, nearly 2,500 miles of ocean will separate him from his family and friends. As a practical matter, Wakinekona may be entirely cut off from his only contacts with the outside world, just as if he had been imprisoned in an institution which prohibited visits by outsiders. Surely the isolation imposed on him by the transfer is far more drastic than that which normally accompanies imprisonment. I cannot agree with the Court that Meachum v. Fano, 427 U. S. 215 (1976), and Montanye v. Haymes, 427 U. S. 236, 243 (1976), compel the conclusion that Wakinekona's transfer implicates no liberty interest. Ante, at 248. Both cases involved transfers of prisoners between institutions located within the same State in which they were convicted, and the Court expressly phrased its holdings in terms of intrastate transfers.[2] Both decisions rested on the premise that no liberty interest is implicated by an initial decision to place a prisoner in one institution in the State rather than another. See Meachum, supra, at 224; Montanye, supra, at 243. On the basis of that premise, the Court concluded that the subsequent transfer of a prisoner to a different facility within the State likewise implicates no liberty interest. In this case, however, we cannot assume that a State's initial placement of an individual in a prison far removed from his family and residence would raise no due process questions. None of our *254 prior decisions has indicated that such a decision would be immune from scrutiny under the Due Process Clause. Actual experience simply does not bear out the Court's assumptions that interstate transfers are routine and that it is 'not unusual' for a prisoner 'to serve practically his entire sentence in a State other than the one in which he was convicted and sentenced.' Ante, at 247. In Hawaii less than three percent of the state prisoners were transferred to prisons in other jurisdictions in 1979, and on a nationwide basis less than one percent of the prisoners held in state institutions were transferred to other jurisdictions.[3] Moreover, the vast majority of state prisoners are held in facilities located less than 250 miles from their homes.[4] Measured against these norms, Wakinekona's transfer to a California prison represents a punishment 'qualitively different from the punishment characteristically suffered by a person convicted of crime.' Vitek v. Jones, supra, at 493. I therefore cannot agree that a State may transfer its prisoners at will, to any place, for any reason, without ever implicating any interest in liberty protected by the Due Process Clause. II Nor can I agree with the majority's conclusion that Hawaii's prison regulations do not create a liberty interest. This Court's prior decisions establish that a liberty interest *255 may be 'created'[5] by state laws, prison rules, regulations, or practices. State laws that impose substantive criteria which limit or guide the discretion of officials have been held to create a protected liberty interest. See, e. g., Hewitt v. Helms, 459 U. S. 460 (1983); Wolff v. McDonnell, 418 U. S. 539 (1974); Greenholtz v. Nebraska Penal Inmates, 442 U. S. 1 (1979); Wright v. Enomoto, 462 F. Supp. 397 (ND Cal. 1976), summarily aff'd, 434 U. S. 1052 (1978). By contrast, a liberty interest is not created by a law which 'imposes no conditions on [prison officials'] discretionary power,' Montanye, supra, at 243, authorizes prison officials to act 'for whatever reason or for no reason at all,' Meachum, supra, at 228, or accords officials 'unfettered discretion,' Connecticut Board of Pardons v. Dumschat, 452 U. S. 458, 466 (1981). The Court misapplies these principles in concluding that Hawaii's prison regulations leave prison officials with unfettered discretion to transfer inmates. Ante, at 249-250. Rule IV establishes a scheme under which inmates are classified upon initial placement in an institution, and must subsequently be reclassified before they can be transferred to another institution. Under the Rule the standard for classifying inmates is their 'optimum placement within the Corrections Division' in light of the 'best interests of the individual, the State, and the community.'[6] In classifying inmates, the Program *256 Committee may not consider punitive aims. It may consider only factors relevant to determining where the individual will be 'best situated,' such as 'his history, his changing needs, the resources and facilities available to the Corrections Divisions, the other inmates/wards, the exigencies of the community, and any other relevant factors.' Paragraph 3 of Rule IV establishes a detailed set of procedures applicable when, as in this case, the reclassification of a prisoner may lead to a transfer involving a 'grievous loss,' a phrase contained in the Rule itself.[7] The procedural rules are cast in mandatory language, and cover such matters as notice, access to information, hearing, confrontation and cross-examination, and the basis on which the Committee is to make its recommendation to the facility administrator. The limitations imposed by Rule IV are at least as substantial as those found sufficient to create a liberty interest in Hewitt v. Helms, supra, decided earlier this Term. In Hewitt an inmate contended that his confinement in administrative custody implicated an interest in liberty protected by the Due Process Clause. State law provided that a prison official could place inmates in administrative custody 'upon his assessment of the situation and the need for control,' or 'where it has been determined that there is a threat of a serious disturbance, or a serious threat to the individual or others,' and mandated certain procedures such as notice and a *257 hearing.[8] This Court construed the phrases ' `the need for control,' or `the threat of a serious disturbance,' ' as 'substantive predicates' which restricted official discretion. Id., at 472. These restrictions, in combination with the mandatory procedural safeguards, 'deman[ded] a conclusion that the State has created a protected liberty interest.' Ibid. Rule IV is not distinguishable in any meaningful respect from the provisions at issue in Helms. The procedural requirements contained in Rule IV are, if anything, far more elaborate than those involved in Helms, and are likewise couched in 'language of an unmistakably mandatory character.' Id., at 471. Moreover, Rule IV, to no less an extent than the state law at issue in Helms, imposes substantive criteria restricting official discretion. In Helms this Court held that a statutory phrase such as 'the need for control' constituted a limitation on the discretion of prison officials to place inmates in administrative custody. In my view Rule IV, which states that transfers are intended to ensure an inmate's 'optimum placement' in accordance with considerations which include 'his changing needs [and] the resources and facilities available to the Corrections Division,' also restricts official discretion in ordering transfers.[9] The Court suggests that, even if the Program Committee does not have unlimited discretion in making recommendations for classifications and transfers, this cannot give rise to a state-created liberty interest because the prison Administrator retains 'completely unfettered' 'discretion to transfer *258 an inmate,' ante, at 249. I disagree. Rule IV, ¶ 3(d)(3), provides for review by the prison Administrator of recommendations forwarded to him by the Program Committee.[10] Even if this provision must be construed as authorizing the Administrator to transfer a prisoner for wholly arbitrary reasons,[11] that mere possibility does not defeat the protectible expectation otherwise created by Hawaii's reclassification and transfer scheme that transfers will take place only if required to ensure an inmate's optimum placement. In Helms a prison regulation also left open the possibility that the Superintendent could decide, for any reason or no reason at all, whether an inmate should be confined in administrative custody.[12] This Court nevertheless held that the state scheme as a whole created an interest in liberty protected by the Due Process Clause. 459 U. S., at 471-472. Helms thus necessarily rejects the view that state laws which impose substantive *259 limitations and elaborate procedural requirements on official conduct create no liberty interest solely because there remains the possibility that an official will act in an arbitrary manner at the end of the process.[13] For the foregoing reasons, I dissent. NOTES [*] Briefs of amici curiae urging reversal were filed for the State of Alaska et al. by Paul L. Douglas, Attorney General of Nebraska, J. Kirk Brown, Assistant Attorney General, Judith W. Rogers, Corporation Counsel of the District of Columbia, and the Attorneys General for their respective jurisdictions as follows: Wilson L. Condon of Alaska, Aviata F. Fa'alevao of American Samoa, Robert K. Corbin of Arizona, Jim Smith of Florida, David H. Leroy of Idaho, William J. Guste, Jr., of Louisiana, William A. Allain of Mississippi, Michael T. Greely of Montana, Richard H. Bryan of Nevada, Irwin I. Kimmelman of New Jersey, Jeff Bingaman of New Mexico, Rufus L. Edmisten of North Carolina, Robert Wefald of North Dakota, William J. Brown of Ohio, Dennis J. Roberts II of Rhode Island, Mark V. Meierhenry of South Dakota, William M. Leech, Jr., of Tennessee, John J. Easton of Vermont, Gerald L. Baliles of Virginia, Kenneth O. Eikenberry of Washington, Chauncey H. Browning of West Virginia, Bronson C. La Follette of Wisconsin, and Steven F. Freudenthal of Wyoming; and for the Commonwealth of Massachusetts et al. by Francis X. Bellotti, Attorney General of Massachusetts, Stephen R. Delinsky, Barbara A. H. Smith, and Leo J. Cushing, Assistant Attorneys General, Anthony Ching, Solicitor General of Arizona, and the Attorneys General for their respective jurisdictions as follows: Wilson L. Condon of Alaska, Aviata F. Fa'alevao of American Samoa, Robert K. Corbin of Arizona, Jim Smith of Florida, David H. Leroy of Idaho, William A. Allain of Mississippi, Michael T. Greely of Montana, Irwin I. Kimmelman of New Jersey, Jeff Bingaman of New Mexico, Rufus L. Edmisten of North Carolina, Robert O. Wefald of North Dakota, William J. Brown of Ohio, Dennis J. Roberts II of Rhode Island, Mark V. Meierhenry of South Dakota, William M. Leech, Jr., of Tennessee, John J. Easton of Vermont, Chauncey H. Browning of West Virginia, and Bronson C. La Follette of Wisconsin. [1] Paragraph 1 of Rule IV states: 'An inmate's . . . classification determines where he is best situated within the Corrections Division. Rather than being concerned with isolated aspects of the individual or punishment (as is the adjustment process), classification is a dynamic process which considers the individual, his history, his changing needs, the resources and facilities available to the Corrections Division, the other inmates . . . , the exigencies of the community, and any other relevant factors. It never inflicts punishment; on the contrary, even the imposition of a stricter classification is intended to be in the best interests of the individual, the State, and the community. In short, classification is a continuing evaluation of each individual to ensure that he is given the optimum placement within the Corrections Division.' App. 20. [2] Petitioners concede, 'for purposes of the argument,' that respondent suffered a 'grievous loss' within the meaning of Rule IV when he was transferred from Hawaii to the mainland. Tr. of Oral Arg. 9, 25. [3] Rule V provides that an inmate may retain legal counsel if his hearing concerns a 'potential Interstate transfer.' App. 25. [4] Respondent also had alleged that the transfer violated the Hawaii Constitution and state regulations and statutes. In light of its dismissal of respondent's federal claims, the District Court declined to exercise pendent jurisdiction over these state-law claims. 459 F. Supp., at 476. [5] Several months before the Court of Appeals handed down its decision, the Supreme Court of Hawaii had held that because Hawaii's prison regulations do not limit the Administrator's discretion to transfer prisoners to the mainland, they do not create any liberty interest. Lono v. Ariyoshi, 63 Haw. 138, 621 P. 2d 976 (1981). In a petition for rehearing in the present case, petitioners directed the Ninth Circuit's attention to the Lono decision. See 664 F. 2d, at 714. The Court of Appeals, however, concluded that the Hawaii court's interpretation of the regulations was not different from its own; the Hawaii court merely had reached a different result on the 'federal question.' The Court of Appeals thus adhered to its resolution of the case. Id., at 714-715. [6] Indeed, in Vitek itself the Court did not read Meachum and Montanye as stating a rule applicable only to intrastate transfers. The Court stated: 'In Meachum v. Fano . . . and Montanye v. Haymes . . . we held that the transfer of a prisoner from one prison to another does not infringe a protected liberty interest.' 445 U. S., at 489 (emphasis added). The Court's other cases describing Meachum and Montanye also have eschewed the narrow reading respondent now proposes. See Hewitt v. Helms, 459 U. S. 460, 467-468 (1983); Moody v. Daggett, 429 U. S. 78, 88, n. 9 (1976). [7] This statute has been invoked to transfer prisoners from Hawaii state facilities to federal prisons on the mainland. See Anthony v. Wilkinson, 637 F. 2d 1130 (CA7 1980), vacated and remanded sub nom. Hawaii v. Mederios, 453 U. S. 902 (1981). [8] After the decisions in Meachum and Montanye, courts almost uniformly have held that an inmate has no entitlement to remain in a prison in his home State. See Beshaw v. Fenton, 635 F. 2d 239, 246-247 (CA3 1980), cert. denied, 453 U. S. 912 (1981); Cofone v. Manson, 594 F. 2d 934, 937, n. 4 (CA2 1979); Sisbarro v. Warden, 592 F. 2d 1, 3 (CA1), cert. denied, 444 U. S. 849 (1979); Fletcher v. Warden, 467 F. Supp. 777, 779-780 (Kan. 1979); Curry-Bey v. Jackson, 422 F. Supp. 926, 931-933 (DC 1976); McDonnell v. United States Attorney General, 420 F. Supp. 217, 220 (ED Ill. 1976); Goodnow v. Perrin, 120 N. H. 669, 671, 421 A. 2d 1008, 1010 (1980); Girouard v. Hogan, 135 Vt. 448, 449-450, 378 A. 2d 105, 106-107 (1977); In re Young, 95 Wash. 2d 216, 227-228, 622 P. 2d 373, 379 (1980); cf. Fajeriak v. McGinnis, 493 F. 2d 468 (CA9 1974) (pre-Meachum transfers from Alaska to other States); Hillen v. Director of Department of Social Services, 455 F. 2d 510 (CA9), cert. denied, 409 U. S. 989 (1972) (pre-Meachum transfer from Hawaii to California). But see In re Young, 95 Wash. 2d, at 233, 622 P. 2d, at 382 (concurring opinion); State ex rel. Olson v. Maxwell, 259 N. W. 2d 621 (N. D. 1977); cf. Tai v. Thompson, 387 F. Supp. 912 (Haw. 1975) (pre-Meachum transfer). [9] Respondent's argument to the contrary is unpersuasive. The Court in Montanye took note that among the hardships that may result from a prison transfer are separation of the inmate from home and family, separation from inmate friends, placement in a new and possibly hostile environment, difficulty in making contact with counsel, and interruption of educational and rehabilitative programs. 427 U. S., at 241, n. 4. These are the same hardships respondent faces as a result of his transfer from Hawaii to California. Respondent attempts to analogize his transfer to banishment in the English sense of 'beyond the seas,' arguing that banishment surely is not within the range of confinement justified by his sentence. But respondent in no sense has been banished; his conviction, not the transfer, deprived him of his right freely to inhabit the State. The fact that his confinement takes place outside Hawaii is merely a fortuitous consequence of the fact that he must be confined, not an additional element of his punishment. See Girouard v. Hogan, 135 Vt., at 449-450, 378 A. 2d, at 106-107. Moreover, respondent has not been exiled; he remains within the United States. In essence, respondent's banishment argument simply restates his claim that a transfer from Hawaii to the mainland is different in kind from other transfers. As has been shown in the text, however, respondent's transfer was authorized by his conviction. A conviction, whether in Hawaii, Alaska, or one of the contiguous 48 States, empowers the State to confine the inmate in any penal institution in any State unless there is state law to the contrary or the reason for confining the inmate in a particular institution is itself constitutionally impermissible. See Montanye, 427 U. S., at 242; id., at 244 (dissenting opinion); Cruz v. Beto, 405 U. S. 319 (1972); Fajeriak v. McGinnis, 493 F. 2d, at 470. [10] In Hewitt v. Helms, 459 U. S. 460 (1983), unlike this case, state law limited the decisionmakers' discretion. To the extent the dissent doubts that the Administrator's discretion under Rule IV is truly unfettered, post, at 258, and n. 11, it doubts the ability or authority of Hawaii Supreme Court to construe state law. [11] In Meachum itself, the Court of Appeals had interpreted the applicable regulations as entitling inmates to a pretransfer hearing, see Fano v. Meachum, 520 F. 2d 374, 379-380 (CA1 1975), but this Court held that state law created no liberty interest. [12] Other courts agree that an expectation of receiving process is not, without more, a liberty interest protected by the Due Process Clause. See, e. g., United States v. Jiles, 658 F. 2d 194, 200 (CA3 1981), cert. denied, 455 U. S. 923 (1982); Bills v. Henderson, 631 F. 2d 1287, 1298-1299 (CA6 1980); Pugliese v. Nelson, 617 F. 2d 916, 924-925 (CA2 1980); Cofone v. Manson, 594 F. 2d, at 938; Lombardo v. Meachum"
"""



from typing import AsyncGenerator, List, Tuple
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
def sample_requests(
    dataset_path: str,
    num_requests: int,
    model_path: str,
    seed=42
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

# 定义一个函数来执行单个请求
def perform_request(session, url, payload, headers):
    start_time = time.perf_counter()
    with session.post(url, json=payload, headers=headers, stream=True) as response:
        # 确保响应成功
        response.raise_for_status()

        # 开始接收streaming响应
        first_token_time = None
        last_token_time = 0
        first_token_inference_time = None
        next_token_inference_time = None
        next_token_time = []
        i = 0
        for line in response.iter_lines():

            token_time = time.perf_counter() - start_time
            if line:  # 忽略心跳
                data = line.decode('utf-8').strip()
                if data.startswith('data: '):
                    data = data[len('data: '):]
                    i = i + 1
                    # print(i, " ", data)
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            choice = json_data['choices'][0]
                            if 'finish_reason' in choice and (choice['finish_reason'] == 'length' or choice['finish_reason'] == 'stop'):
                                if 'first_token_time' in choice and isinstance(choice['first_token_time'], float):
                                    first_token_inference_time = choice['first_token_time']
                                if 'rest_token_time' in choice and isinstance(choice['rest_token_time'], float):
                                    next_token_inference_time = choice['rest_token_time']
                            else:
                                # 记录第一个token的时间
                                if first_token_time is None:
                                    first_token_time = token_time
                                else:
                                    # 记录后续token的时间
                                    next_token_time.append(token_time - last_token_time)
                                last_token_time = token_time
                    except json.JSONDecodeError:
                        pass  # 如果不是JSON数据，忽略错误

        # 返回第一个token和后续token的latency
        end_time = time.perf_counter()
        # print("length: ", len(next_token_time))

        return first_token_time, np.mean(next_token_time), end_time - start_time, first_token_inference_time, next_token_inference_time

def extend_list_to_length(lst, target_length):
    if target_length <= len(lst):
        return lst[:]

    # 计算每个元素需要复制的次数
    times = target_length // len(lst)
    # 计算不能整除的剩余部分
    remainder = target_length % len(lst)

    # 生成新列表：重复整个列表times次，再加上前remainder个元素
    extended_list = lst * times + lst[:remainder]

    return extended_list

# 定义一个函数来执行benchmark
def benchmark(llm_urls, model, prompt, num_requests, max_concurrent_requests, max_tokens, is_warmup=False, dataset=None):
    # 定义请求的payload和headers

    headers = {"Content-Type": "application/json"}

    first_token_latencies = []
    next_token_latencies = []
    total_responce_times = []
    first_token_inference_times = []
    next_token_inference_times = []
    cur_url_index = 0
    sampled_requests = []
    prompt_token_lens = []
    output_tokens_lens = []


    if not dataset is None:
        sampled_requests = sample_requests(dataset, num_requests, model)

    # 使用Session对象以便复用连接
    with requests.Session() as session:
        # 创建一个线程池
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # 开始计时
            # time.sleep(1)
            llm_url = llm_urls[cur_url_index]
            cur_url_index = (cur_url_index + 1) % len(llm_urls)

            cur_llm_urls = extend_list_to_length(llm_urls, max_concurrent_requests)
            cur_len = len(cur_llm_urls)
            if dataset is None:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "ignore_eos": True,
                    "stream": True  # 开启streaming模式
                }
                futures = [executor.submit(perform_request, session, cur_llm_urls[index % cur_len], payload, headers) for index in range(num_requests)]
            else:
                payloads = []
                for index in range(num_requests):
                    prompt, prompt_len, output_len = sampled_requests[index]
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "n": 1,
                        "best_of": 1,
                        "use_beam_search": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": output_len,
                        "ignore_eos": True,
                        "stream": True  # 开启streaming模式
                    }
                    prompt_token_lens.append(prompt_len)
                    output_tokens_lens.append(output_len)
                    payloads.append(payload)
                futures = [executor.submit(perform_request, session, cur_llm_urls[index % cur_len], payloads[index], headers) for index in range(num_requests)]


            start_time = time.perf_counter()

            if is_warmup:
                phase = "Warm Up"
            else:
                phase = "Benchmarking"
            with tqdm(total=num_requests, desc=phase, unit="req", ncols=100) as pbar:
                # 等待所有请求完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        first_token_latency, next_token_latency, total_responce_time, first_token_inference_time, next_token_inference_time = future.result()
                        first_token_latencies.append(first_token_latency)
                        next_token_latencies.append(next_token_latency)
                        total_responce_times.append(total_responce_time)
                        if first_token_inference_time:
                            first_token_inference_times.append(first_token_inference_time)
                        if next_token_inference_time:
                            next_token_inference_times.append(next_token_inference_time)
                    except Exception as e:
                        print(f"Request failed: {e}")
                    pbar.update(1)

            # 计算总用时
            if is_warmup:
                return
            total_time = time.perf_counter() - start_time
            print(f"Total time for {num_requests} requests with {max_concurrent_requests} concurrent requests: {total_time} seconds.")
            print(f"Average responce time: {np.mean(total_responce_times)}")
            if dataset is None:
                print(f"Token throughput: {num_requests * max_tokens / total_time}")
            else:
                print(f"Output token throughput: {sum(output_tokens_lens) / total_time}")
                print(f"Total token throughput: {(sum(prompt_token_lens) + sum(output_tokens_lens)) / total_time}")
            print()
            if first_token_latencies:
                average_first_token_latency = sum(first_token_latencies) / len(first_token_latencies)
                p90_first_token_latency = np.percentile(first_token_latencies, 90)
                p95_first_token_latency = np.percentile(first_token_latencies, 95)
                average_first_token_inference_latency = np.mean(first_token_inference_times)
                print(f"Average first token latency: {average_first_token_latency * 1000} milliseconds.")
                print(f"P90 first token latency: {p90_first_token_latency * 1000} milliseconds.")
                print(f"P95 first token latency: {p95_first_token_latency * 1000} milliseconds.")
                #print(f"Average first token inference latency: {average_first_token_inference_latency * 1000} milliseconds.")
                print()
            if next_token_latencies:
                average_next_token_latency = sum(next_token_latencies) / len(next_token_latencies)
                p90_next_token_latency = np.percentile(next_token_latencies, 90)
                p95_next_token_latency = np.percentile(next_token_latencies, 95)
                average_next_token_inference_latency = np.mean(next_token_inference_times)
                print(f"Average next token latency: {average_next_token_latency * 1000} milliseconds.")
                print(f"P90 next token latency: {p90_next_token_latency * 1000} milliseconds.")
                print(f"P95 next token latency: {p95_next_token_latency * 1000} milliseconds.")
                #print(f"Average next token inference latency: {average_next_token_inference_latency * 1000} milliseconds.")
                print()


# 设置benchmark参数
LLM_URLS = [f"http://localhost:{PORT}/v1/completions" for PORT in [8000]]

MODEL = "/llm/models/" + model_name
MAX_TOKENS = output_length  # 修改 MAX_TOKENS 为 output_length

# if "Qwen" not in MODEL and "chatglm" not in MODEL:
#     print("using Llama PROMPT")
#     PROMPT = ENGLISH_PROMPT
# else:
#     print("using Qwen/chatglm PROMPT")
#     PROMPT = CHINESE_PROMPT

PROMPT = ENGLISH_PROMPT

# 加载模型的 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
# print("old input_ids.shape:"+ str(input_ids.shape))

# 限制输入长度为 input_length
input_ids = input_ids[:, :input_length]
# print("latest input_ids.shape:"+ str(input_ids.shape))

# 将截断后的 prompt 解码回来
true_str = tokenizer.batch_decode(input_ids)[0]
PROMPT = true_str

max_batch=int(max_seq)

for MAX_CONCURRENT_REQUESTS in [max_batch]:
    NUM_WARMUP = 2 * MAX_CONCURRENT_REQUESTS
    NUM_REQUESTS = 4 * MAX_CONCURRENT_REQUESTS  # 总请求次数

    # to avoid warm_up time out
    benchmark(LLM_URLS, MODEL, PROMPT_1024, 2, 1, 32, is_warmup = True)
    benchmark(LLM_URLS, MODEL, PROMPT, NUM_WARMUP, MAX_CONCURRENT_REQUESTS, MAX_TOKENS, is_warmup = True)

    # 运行benchmark
    benchmark(LLM_URLS, MODEL, PROMPT, NUM_REQUESTS, MAX_CONCURRENT_REQUESTS, MAX_TOKENS)
