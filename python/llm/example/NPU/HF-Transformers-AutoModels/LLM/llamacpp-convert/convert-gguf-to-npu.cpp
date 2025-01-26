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

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -o output_dir --low-bit sym_int4 --quantization-group-size 0\n", argv[0]);
    LOG("\n");
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

    std::string output_dir = params.npu_outfile;
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

    return 0;
}
