//
// Copyright 2016 The BigDL Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "npu_common.h"
#include "npu_llm.h"


static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m npu_model_dir [-cnv] [-n n_predict] [prompt]\n", argv[0]);
    printf("\n");
}


const std::string llama2_template = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\n%s [/INST]";
const std::string llama3_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
const std::string minicpm_template = "<用户>%s<AI>";
const std::string qwen2_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
const std::string qwen2_deepseek_template = "<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>%s<｜Assistant｜>";


std::string add_chat_history(npu_model_params model_params,
                             std::string new_prompt, std::string chat_history, bool is_input) {
    char prompt[8092];
    if (model_params.model_type == std::string("llama") && model_params.vocab_size == 32000) {
        if (chat_history == ""){
            sprintf_s(prompt, llama2_template.c_str(), new_prompt.c_str());
        }else{
            if (is_input){
                std::string input_template = "%s%s [/INST]";
                sprintf_s(prompt, input_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
            else{
                std::string res_template = "%s%s </s><s>[INST]";
                sprintf_s(prompt, res_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
        }
        
    } else if (model_params.model_type == std::string("llama")) {
        if (chat_history == ""){
            sprintf_s(prompt, llama3_template.c_str(), new_prompt.c_str());
        }else{
            if (is_input){
                std::string input_template = "%s<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>";
                sprintf_s(prompt, input_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
            else{
                std::string res_template = "%s<|start_header_id|>assistant<|end_header_id|>\n\n%s<|eot_id|>";
                sprintf_s(prompt, res_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
        }
    } else if (model_params.model_type == std::string("qwen2") && model_params.max_position_embeddings == 131072) {
        // For DeepSeek-R1
        if (chat_history == ""){
            sprintf_s(prompt, qwen2_deepseek_template.c_str(), new_prompt.c_str());
        }else{
            if (is_input){
                std::string input_template = "%s%s<｜Assistant｜>";
                sprintf_s(prompt, input_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
            else{
                std::string res_template = "%s%s<｜User｜>";
                sprintf_s(prompt, res_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
        }
    } else if (model_params.model_type == std::string("qwen2")) {
        if (chat_history == ""){
            sprintf_s(prompt, qwen2_template.c_str(), new_prompt.c_str());
        }else{
            if (is_input){
                std::string input_template = "%s%s<|im_end|>\n<|im_start|>assistant";
                sprintf_s(prompt, input_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
            else{
                std::string res_template = "%s%s<|im_end|>\n<|im_start|>user\n";
                sprintf_s(prompt, res_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
        }
    } else if (model_params.model_type == std::string("minicpm")) {
        if (chat_history == ""){
            sprintf_s(prompt, minicpm_template.c_str(), new_prompt.c_str());
        }else{
            if (is_input){
                std::string input_template = "%s%s<AI>";
                sprintf_s(prompt, input_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
            else{
                std::string res_template = "%s%s<用户>";
                sprintf_s(prompt, res_template.c_str(), chat_history.c_str(), new_prompt.c_str());
            }
        }
    } else {
        sprintf_s(prompt, chat_history.c_str(), new_prompt.c_str());
    }
    return prompt;
}

std::string run_generate(void* void_model, int32_t* embd_inp_ptr, int32_t embd_inp_size,
                         npu_model_params model_params, tokenizer_params tok_params, npu_generation_params generation_params){
    float* logits = run_prefill(void_model, embd_inp_ptr, embd_inp_size,
                                generation_params.repetition_penalty);
    int32_t token = llm_sample_token(logits, true, model_params.vocab_size);

    std::vector<int32_t> embd;  // output ids
    embd.push_back(token);

    int token_nums = 0;
    for (int i = 1; i < generation_params.max_new_token; i++){
        auto logits = run_decode(void_model, embd[i-1],
                                 generation_params.repetition_penalty);
        int32_t token = llm_sample_token(logits, true, model_params.vocab_size);
        if (std::find(tok_params.eos_token_id.begin(), tok_params.eos_token_id.end(), token) == tok_params.eos_token_id.end()){
            embd.push_back(token);
            token_nums ++;
        } else {
            break;
        }
    }

    std::string output = llm_decode(embd);

    return output;
}


int main(int argc, char ** argv) {
    common_params params;

    // path to the npu model directory
    char* model_dir;
    // prompt to generate text from
    std::string prompt = "AI是什么?";
    // number of tokens to predict
    int n_predict = 32;
    bool cnv_mode = false;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_dir = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-cnv") == 0){
                // multi-round conversation mode
                cnv_mode = true;
                break;
            }else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_dir == nullptr || model_dir[0] == '\0') {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    params.n_predict = n_predict;
    params.model = model_dir;
    params.prompt = prompt;

    // npu_model_params model_params;
    void* model = load_model_from_file(params.model);
    npu_model_params model_params;
    load_config_from_file(model_params, params.model);

    tokenizer_params tok_params;
    load_tokenizer(tok_params, params.model);

    npu_generation_params generation_params;
    load_generation_config_from_file(generation_params, params.model);
    generation_params.max_new_token = n_predict;

    if (cnv_mode){
        std::string prompt;
        std::string history = "";
        std::string response;
        while(true){
            std::cout << "User:";
            std::getline(std::cin, prompt);
            if (prompt == "exit"){
                break;
            }
            else{
                // process prompt with chat history
                std::string full_prompt = add_chat_history(model_params, prompt, history, true);

                // tokenize input
                std::vector<int32_t> embd_inp = llm_tokenize(full_prompt, false);
                if (embd_inp.size() > model_params.max_prompt_len){
                    // empty chat history
                    full_prompt = add_chat_history(model_params, prompt, "", true);
                    embd_inp = llm_tokenize(full_prompt, false);
                }

                generation_params.max_new_token = model_params.kv_len - embd_inp.size();
                
                response = run_generate(model, embd_inp.data(), embd_inp.size(),
                                        model_params, tok_params, generation_params);

                std::cout << "Assistant:";
                std::cout << response << std::endl;

                history = add_chat_history(model_params, response, full_prompt, false);

                reset(model);
            }
        }
    }
    else{
        std::string full_prompt = add_chat_template(model_params, params.prompt);

        // tokenize input
        std::vector<int32_t> embd_inp = llm_tokenize(full_prompt, false);

        // single text generation
        std::string output = run_generate(model, embd_inp.data(), embd_inp.size(),
                                          model_params, tok_params, generation_params);

        std::cout << output << std::endl << std::endl;
        llm_perf_print(model);
    }
    return 0;
}