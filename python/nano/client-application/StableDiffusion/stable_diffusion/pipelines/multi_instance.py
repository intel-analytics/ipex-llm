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

def multi_instance_generate(pipe, prompts, save_paths, result_queue, batch_size=4, **kwargs):
    if len(prompts) == 0:
        result_queue.put([])
    else:
        generated_image_list = []
        prompt_list = list(divide_chunks(prompts, batch_size))
        if save_paths is not None:
            save_paths = list(divide_chunks(save_paths, batch_size))
        else:
            save_paths = [None] * len(prompt_list)
        
        for p, sp in zip(prompt_list, save_paths):
            imgs = pipe.generate(prompt=p, **kwargs)
            if hasattr(imgs, "images"):
                imgs = imgs.images
            
            if sp is not None:
                img_per_prompt = kwargs.get("num_images_per_prompt", 1)
                if img_per_prompt > 1:
                    sp = get_new_path_list(sp, img_per_prompt)
                
                for img, p in zip(imgs, sp):
                    img.save(p)
            else:
                generated_image_list.extend(imgs)
        
        result_queue.put(generated_image_list)

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]
    
def get_new_path_list(save_paths, num_images_per_prompt):
    new_save_paths = []
    for p in save_paths:
        parts = p.split(".")
        new_save_paths.extend([f"{parts[0]}_{id}.{parts[1]}" for id in range(num_images_per_prompt)])
    return new_save_paths
    