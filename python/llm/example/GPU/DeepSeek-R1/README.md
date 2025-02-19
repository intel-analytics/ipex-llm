# Run DeepSeek V3/R1 on Intel CPU and GPUs using IPEX-LLM

```bash
# From docker image: intelanalytics/ipex-llm-serving-xpu:2.2.0-b13
pip install transformers==4.48.3 accelerate trl==0.12.0

python generate_hybrid.py --load-path /llm/models/DeepSeek-R1-int4-1/ --n-predict 128
```

Sample output:
```
-------------------- Prompt --------------------

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question: If \( a > 1 \), then the sum of the real solutions of \( \sqrt{a} - \sqrt{a + x} = x \) is equal to:. Assistant: <think>

-------------------- Output --------------------

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Question: If \( a > 1 \), then the sum of the real solutions of \( \sqrt{a} - \sqrt{a + x} = x \) is equal to:. Assistant: <think>
Okay, let's see. The problem is asking for the sum of the real solutions of the equation √a - √(a + x) = x, given that a > 1. Hmm, I need to find the values of x that satisfy this equation and then add them up. 

First, I should probably try to solve for x. The equation has square roots, so maybe squaring both sides will help eliminate them. But I have to be careful because squaring can sometimes introduce extraneous solutions. Let me start by writing down the equation again:

√a - √(a + x) = x
```