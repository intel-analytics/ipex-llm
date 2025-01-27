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

#include "common.h"
#include "npu_llm.h"

#include "llamacpp/arg.h"
#include "llamacpp/common.h"
#include "llamacpp/log.h"
#include "llamacpp/llama.h"
#include <filesystem>
#include <vector>
#include<iostream>

#ifdef _WIN32
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif


struct gguf_tokenizer_params {
    llama_context * ctx;
    int32_t bos_token_id;
    int32_t eos_token_id;
    bool add_bos;
    bool parse_special;
};


static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m npu_model_dir [-cnv] [-n n_predict] [prompt]\n", argv[0]);
    printf("\n");
}


vector<int32_t> gguf_tokenize(std::string prompt,
                              gguf_tokenizer_params tok_params) {
  int n_tokens = prompt.length() + 2 * tok_params.add_bos;   // TODO: no need
  std::vector<int32_t> ids = llama_tokenize(tok_params.ctx, prompt,
                                            tok_params.add_bos, tok_params.parse_special);
  return ids;
}

std::string gguf_decode(vector<int32_t> tokens, gguf_tokenizer_params tok_params) {
  std::string output = llama_detokenize(tok_params.ctx, tokens, tok_params.parse_special);
  return output;
}


const std::string llama2_template = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\n%s [/INST]";
const std::string llama3_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
const std::string minicpm_template = "<用户>%s<AI>";
const std::string qwen2_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";


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
                         npu_model_params model_params, gguf_tokenizer_params tok_params, npu_generation_params generation_params){
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
        if (tok_params.eos_token_id != token){
            embd.push_back(token);
            token_nums ++;
        } else {
            break;
        }
    }

    std::string output = gguf_decode(embd, tok_params);

    return output;
}


int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_NPU, print_usage)) {
        return 1;
    }

    gpt_init();

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    enum gguf_npu_qtype type;

    if (params.low_bit == "sym_int4") {
        type = GGUF_TYPE_NPU_CW_Q4_0;
    } else if (params.low_bit == "asym_int4") {
        type = GGUF_TYPE_NPU_CW_Q4_1;
    } else {
        std::cerr << "\033[31m" << __func__ << ": error: Only support sym_int4 and asym_int4 but got " << params.low_bit << "\033[0m\n" << std::endl;
        exit(1);
    }

    if (params.npu_outfile == "NPU_MODEL") {
        fprintf(stderr , "\033[31m%s: error: Please provide npu model output dir with -o <output_dir>\033[0m\n" , __func__);
        exit(1);
    }

    // initialize the model

    llama_model_params model_params = llama_model_params_from_gpt_params(params);

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const std::string output_dir = params.npu_outfile;
    std::filesystem::path dirPath = output_dir;

    // handle weight first
    if(std::filesystem::create_directory(dirPath)) {
        std::cout << "Directory created: " << dirPath << std::endl;
    } else {
        std::cout << "Failed to create directory or already exists: " << dirPath << "\n";
    }

    std::string weight_path = output_dir + PATH_SEP + "model_weights"; // TODO: optimize /
    dirPath = weight_path;
    if(std::filesystem::create_directory(dirPath)) {
        std::cout << "Directory created: " << dirPath << std::endl;
    } else {
        std::cout << "Failed to create directory or already exists: " << dirPath << "\n";
    }

    if (params.quantization_group_size != 0) {
        std::cerr << "\033[31mOnly support quantization group_size=0, fall back to channel wise quantization.\033[0m\n" << std::endl;
    }

    std::cout << "\033[32mConverting GGUF model to " <<  params.low_bit << " NPU model...\033[0m" << std::endl;
    convert_gguf_to_npu_weight(model, weight_path.c_str(), type);

    std::cout << "\033[32mModel weights saved to " << weight_path << "\033[0m"<< std::endl;

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    common_params npu_common_params;
    npu_common_params.n_predict = params.n_predict;
    npu_common_params.model = const_cast<char*>(output_dir.c_str());
    npu_common_params.prompt = params.prompt;
    // npu_model_params model_params;
    void* npu_model = load_model_from_file(npu_common_params.model);
    npu_model_params model_params_npu;
    load_config_from_file(model_params_npu, npu_common_params.model);

    // load gguf model again with vocab_only for tokenizer
    llama_backend_init();
    llama_model_params gguf_model_params = llama_model_default_params();
    gguf_model_params.vocab_only = true;
    llama_model * gguf_model = llama_load_model_from_file(params.model.c_str(), gguf_model_params);
    auto cparams = llama_context_default_params();
    llama_context * vocab_ctx = llama_new_context_with_model(gguf_model, cparams);
    // if (!gguf_model) {
    //     fprintf(stderr, "Error: could not load model from file '%s'.\n", gguf_path);
    //     return 1;
    // }

    gguf_tokenizer_params tok_params;
    tok_params.ctx = vocab_ctx;
    tok_params.bos_token_id =  llama_token_bos(gguf_model);
    tok_params.eos_token_id =  llama_token_eos(gguf_model);
    tok_params.add_bos = llama_add_bos_token(gguf_model);
    tok_params.parse_special = true;

    // tokenizer_params tok_params;
    // load_tokenizer(tok_params, npu_common_params.model);

    npu_generation_params generation_params;
    load_generation_config_from_file(generation_params, npu_common_params.model);
    generation_params.max_new_token = npu_common_params.n_predict;

    if (params.interactive){
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
                std::string full_prompt = add_chat_history(model_params_npu, prompt, history, true);

                // tokenize input
                std::vector<int32_t> embd_inp = gguf_tokenize(full_prompt, tok_params);
                if (embd_inp.size() > model_params_npu.max_prompt_len){
                    // empty chat history
                    full_prompt = add_chat_history(model_params_npu, prompt, "", true);
                    embd_inp = gguf_tokenize(full_prompt, tok_params);
                }

                generation_params.max_new_token = model_params_npu.kv_len - embd_inp.size();
                
                response = run_generate(npu_model, embd_inp.data(), embd_inp.size(),
                                        model_params_npu, tok_params, generation_params);

                std::cout << "Assistant:";
                std::cout << response << std::endl;

                history = add_chat_history(model_params_npu, response, full_prompt, false);

                reset(npu_model);
            }
        }
    }
    else{
        std::string full_prompt = add_chat_template(model_params_npu, params.prompt);

        // tokenize input
        std::vector<int32_t> embd_inp = gguf_tokenize(full_prompt, tok_params);

        // single text generation
        std::string output = run_generate(npu_model, embd_inp.data(), embd_inp.size(),
                                          model_params_npu, tok_params, generation_params);

        std::cout << output << std::endl << std::endl;
        llm_perf_print(npu_model);
    }

    llama_free(vocab_ctx);
    llama_free_model(gguf_model);

    llama_backend_free();
    return 0;
}