#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "common.h"
#include "npu_llm.h"


static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m npu_model_dir [-n n_predict] [prompt]\n", argv[0]);
    printf("\n");
}


int main(int argc, char ** argv) {
    common_params params;

    // path to the npu model directory
    std::string model_dir;
    // prompt to generate text from
    std::string prompt = "AI是什么?";
    // number of tokens to predict
    int n_predict = 32;

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
            } else if (strcmp(argv[i], "-n") == 0) {
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
        if (model_dir.empty()) {
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

    npu_model_params model_params;
    NPUModel* model = load_model_from_file(model_params, params.model);

    tokenizer_params tok_params;
    load_tokenizer(tok_params, params.model);

    std::string full_prompt = add_chat_template(model_params, params.prompt);
    std::cout << "Input: " << std::endl;
    std::cout << full_prompt << std::endl;

    // tokenize input
    std::vector<int32_t> embd_inp = llm_tokenize(full_prompt, false);

    std::vector<int32_t> embd;  // output ids
    auto start = std::chrono::high_resolution_clock::now();
    float* logits = run_prefill(model, embd_inp);
    int32_t token = llm_sample_token(logits, true, model_params);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("\nPrefill %d tokens cost %d ms.\n", embd_inp.size(), duration.count());
    embd.push_back(token);

    int token_nums = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < params.n_predict; i++){
        auto logits = run_decode(model, embd[i-1]);
        int32_t token = llm_sample_token(logits, true, model_params);
        if (token != tok_params.eos_token_id) {
            embd.push_back(token);
            token_nums ++;
        } else {
            break;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::string output = llm_decode(embd);

    std::cout << "Output: " << std::endl;
    std::cout << output << std::endl;

    printf("\nDecode %d tokens cost %d ms (avg %f ms each token).\n", token_nums, duration.count(), (float)duration.count() / token_nums);

    return 0;
}