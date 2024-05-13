# init llama-cpp first
mkdir -p /llm/llama-cpp
cd /llm/llama-cpp
init-llama-cpp

# change the model_path to run
if [[ "$DEVICE" == "Arc" || "$DEVICE" == "ARC" ]]; then
    source ipex-llm-init -g --device Arc
    python run.py
elif [[ "$DEVICE" == "Flex" || "$DEVICE" == "FLEX" ]]; then
    source ipex-llm-init -g --device Flex
    python run.py
elif [[ "$DEVICE" == "Max" || "$DEVICE" == "MAX" ]]; then
    source ipex-llm-init -g --device Max
    python run.py
else
    echo "Invalid DEVICE specified."
fi
model="/models/"$bench_model

promt_32_32="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
# prompt is too long (998 tokens, max 508)
# promt_1024_128="The sun was setting over the horizon, casting long shadows across the dusty ground of the town square. The last rays of light streamed through the gaps between the buildings, illuminating the cobblestones and the people milling about. A group of children played a rough game of tag, their laughter filling the air. In the center of the square stood a lone figure, a man with a tired face and weary eyes. He was tall and broad shouldered, but his posture was slumped and his head hung low. Despite the warmth of the day, he wore a thick woolen coat that seemed too heavy for the weather. The man looked out at the crowd, his gaze sweeping over the faces of the townspeople as they went about their business. His eyes settled on a young woman standing on the edge of the square, watching him with curiosity. She was pretty, with chestnut hair pulled back into a ponytail and bright green eyes that sparkled in the fading light. The man felt a sudden jolt of recognition, as if he had seen her before. But he couldn't remember where or when. He tried to shake off the feeling, but it lingered like a ghostly presence. Suddenly, the woman's eyes widened in alarm, and she began to run towards him. Her movements were urgent and panicked, as if she was trying to escape something or someone. The man watched her go, his confusion growing with each step. He turned to look around the square, but there was no sign of anyone else. It was as if the woman had appeared out of nowhere, and now she was gone just as suddenly. The man rubbed his temples, feeling a mounting sense of unease. He tried to make sense of what he had just seen, but it was like trying to grasp smoke in his hands. He shook his head, frustrated with himself for being so easily spooked. Just then, a voice called out from behind him. \"Hey there! You look lost.\" The man turned to see a young man standing behind him, a friendly smile on his face. He was tall and lean, with tousled blond hair that seemed to glow in the fading light. His eyes were bright and curious, as if he was eager to know everything about the world around him. The man hesitated for a moment, unsure of how to respond. He wasn't used to talking to strangers, especially ones who looked so young and innocent. But there was something about the boy that made him feel comfortable, as if he had known him all his life. \"I'm not lost,\" he said finally, his voice gruff but friendly. \"Just a little confused, I guess.\" The boy grinned. \"Well, I can help with that! My name is Jake, by the way. What's your name?\" The man hesitated for a moment, then introduced himself as Michael. They chatted for a few minutes, exchanging small talk and pleasantries. But even as they spoke, the man couldn't shake the feeling that something was amiss. It was as if he had forgotten something important, something that he needed to remember before it was too late. He excused himself from Jake, promising to come back later and chat some more. As he walked away, he felt a growing sense of unease. Something wasn't right, and he needed to figure out what it was before it was too late. [CHAPTER 4: THE LIBRARY](9781441125608_epub_itb-ch4.xhtml) The man returned to the library, his mind still racing with thoughts of the mysterious boy and the strange feeling that had been nagging at him all day. He wandered through the shelves, scanning the titles of books and flipping through their pages, searching for something that might help him remember what he had forgotten. It wasn't until he stumbled upon a book on ancient myths and legends that he felt a spark of recognition. As he read through the stories of gods and monsters, he began to recall fragments of memories from his own life. Memories of strange symbols carved into walls, of dreams filled with images of a dark forest and a mysterious figure. He realized with a start that these were not just random memories, but pieces of a larger puzzle that had been scattered throughout his life. He felt a sudden urgency to put them together before they faded away completely. As he continued to read through the book, he began to notice patterns and connections between the myths and his own memories. The symbols he had seen as a child were not just random carvings, but part of an ancient language that held the key to unlocking the secrets of his past. With newfound determination..."

# warm-up
./main -m $model -n 32 --prompt "${promt_32_32}"  -t 8 -e -ngl 999 --color
./main -m $model -n 32 --prompt "${promt_32_32}"  -t 8 -e -ngl 999 --color
#./main -m $model -n 128 -t 8 -e -ngl 999 --color --prompt "${promt_1024_128}"
