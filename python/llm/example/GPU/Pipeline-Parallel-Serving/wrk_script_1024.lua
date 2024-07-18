wrk.method = "POST"
wrk.headers["accept"] = "application/json"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"inputs": "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. However, her parents were always telling her to stay close to home, to be careful, and to avoid any danger. But the little girl was stubborn, and she wanted to see what was on the other side of the mountain. So she sneaked out of the house one night, leaving a note for her parents, and set off on her journey. As she climbed the mountain, the little girl felt a sense of excitement and wonder. She had never been this far away from home before, and she couldnt wait to see what she would find on the other side. She climbed higher and higher, her lungs burning from the thin air, until she finally reached the top of the mountain. And there, she found a beautiful meadow filled with wildflowers and a sparkling stream. The little girl danced and played in the meadow, feeling free and alive. She knew she had to return home eventually, but for now, she was content to enjoy her adventure. As the sun began to set, the little girl reluctantly made her way back down the mountain, but she knew that she would never forget her adventure and the joy of discovering something new and exciting. And whenever she felt scared or unsure, she would remember the thrill of climbing the mountain and the beauty of the meadow on the other side, and she would know that she could face any challenge that came her way, with courage and determination. She carried the memories of her journey in her heart, a constant reminder of the strength she possessed. The little girl returned home to her worried parents, who had discovered her note and anxiously awaited her arrival. They scolded her for disobeying their instructions and venturing into the unknown. But as they looked into her sparkling eyes and saw the glow on her face, their anger softened. They realized that their little girl had grown, that she had experienced something extraordinary. The little girl shared her tales of the mountain and the meadow with her parents, painting vivid pictures with her words. She spoke of the breathtaking view from the mountaintop, where the world seemed to stretch endlessly before her. She described the delicate petals of the wildflowers, vibrant hues that danced in the gentle breeze. And she recounted the soothing melody of the sparkling stream, its waters reflecting the golden rays of the setting sun. Her parents listened intently, captivated by her story. They realized that their daughter had discovered a part of herself on that journey—a spirit of curiosity and a thirst for exploration. They saw that she had learned valuable lessons about independence, resilience, and the beauty that lies beyond ones comfort zone. From that day forward, the little girls parents encouraged her to pursue her dreams and embrace new experiences. They understood that while there were risks in the world, there were also rewards waiting to be discovered. They supported her as she continued to embark on adventures, always reminding her to stay safe but never stifling her spirit. As the years passed, the little girl grew into a remarkable woman, fearlessly exploring the world and making a difference wherever she went. The lessons she had learned on that fateful journey stayed with her, guiding her through challenges and inspiring her to live life to the fullest. And so, the once timid little girl became a symbol of courage and resilience, a reminder to all who knew her that the greatest joys in life often lie just beyond the mountains we fear to climb. Her story spread far and wide, inspiring others to embrace their own journeys and discover the wonders that awaited them. In the end, the little girls adventure became a timeless tale, passed down through generations, reminding us all that sometimes, the greatest rewards come to those who dare to step into the unknown and follow their hearts. With each passing day, the little girls story continued to inspire countless individuals, igniting a spark within their souls and encouraging them to embark on their own extraordinary adventures. The tale of her bravery and determination resonated deeply with people from all walks of life, reminding them of the limitless possibilities that awaited them beyond the boundaries of their comfort zones. People marveled at the little girls unwavering spirit and her unwavering belief in the power of dreams. They saw themselves reflected in her journey, finding solace in the knowledge that they too could overcome their fears and pursue their passions. The little girl\'s story became a beacon of hope, a testament to the human spirit",  "parameters": {"max_new_tokens": 128, "min_new_tokens": 128}}'

logfile = io.open("wrk.log", "w");

response = function(status, header, body)
     logfile:write("status:" .. status .. "\n" .. body .. "\n-------------------------------------------------\n");
end
