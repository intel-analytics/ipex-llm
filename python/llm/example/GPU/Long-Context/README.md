# Running Long-Context generation using IPEX-LLM on Intel Arc™ A770 Graphics

Long-Context Generation is a critical aspect in various applications, such as document summarization, extended conversation handling, and complex question answering. Effective long-context generation can lead to more coherent and contextually relevant responses, enhancing user experience and model utility.

This folder contains examples of running long-context generation with IPEX-LLM on Intel Arc™ A770 Graphics(16GB GPU memory):

<!-- TODO: Maybe like this after adding more examples:
- [Single GPU](Single GPU): single GPU examples w & w/o batch.
- [Multiple GPU](Multiple GPU): multiple GPU examples w & w/o batch. -->
- [LLaMA2-32K](LLaMA2-32K): examples of running LLaMA2-32K models with INT4/FP8 precision.
- [ChatGLM3-32K](Chatglm3-32K): examples of running ChatGLM3-32K models with INT4/FP8 precision.

### Maximum Input Length for Different Models with INT4/FP8 Precision.

- **INT4**

    | Model Name | Low Memory Mode | Maximum Input Length | Output Length |
    | -- | -- | -- | -- |
    | LLaMA2-7B-32K | Disable | 10K | 512 |
    |  | Enable | 12K | 512 |
    | ChatGLM3-6B-32K | Disable | 9K | 512 |
    |  | Enable | 10K | 512 |

- **FP8**

    | Model Name | Low Memory Mode | Maximum Input Length | Output Length |
    | -- | -- | -- | -- |
    | LLaMA2-7B-32K | Disable | 7K | 512 |
    |  | Enable | 9K | 512 |
    | ChatGLM3-6B-32K | Disable | 8K | 512|
    |  | Enable | 9K | 512 |

> Note: If you need to run longer input or use less memory, please set `IPEX_LLM_LOW_MEM=1` to enable **low memory mode**, which will enable memory optimization and may slightly affect the performance.