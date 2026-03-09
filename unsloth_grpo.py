import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

# required patch to get grpo working with unsloth
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 1024      # longer helps with reasoning chains
dtype = None               # auto-detects bfloat16 on a100
load_in_4bit = True        # 4bit keeps memory low with barely any accuracy loss

# qwen2.5 is really good for math/gsm8k
model_name = "Qwen/Qwen2.5-7B-Instruct"
# alternatives: "meta-llama/Llama-3.1-8B-Instruct" or "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",           # needed for gated models
)

# adding lora adapters - unsloth's optimized version
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                    # higher rank = better quality but slower
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # important for keeping memory in check
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

dataset = load_dataset("openai/gsm8k", "main", split = "train")

# format prompts for grpo - step by step reasoning with boxed answer as ground truth
def formatting_prompts_func(examples):
    questions = examples["question"]
    answers   = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        text = f"Question: {question}\n\nLet's think step by step."
        texts.append(text)
    return { "prompt" : texts, "ground_truth" : [a.split("####")[-1].strip() for a in answers] }

dataset = dataset.map(formatting_prompts_func, batched = True)

import re

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{(.*?)\}', text)
    return match.group(1).strip() if match else None

def accuracy_reward_func(completions, ground_truths, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred = extract_boxed_answer(completion)
        reward = 1.0 if pred == gt else 0.0
        rewards.append(reward)
    return rewards

def format_reward_func(completions, **kwargs):
    # bonus if the response has \boxed{} and actual reasoning steps
    return [1.0 if "\\boxed{" in c and len(c) > 200 else 0.5 for c in completions]

reward_funcs = [accuracy_reward_func, format_reward_func]

# grpo config, tuned for a100 speed and memory
training_args = GRPOConfig(
    output_dir = "outputs/gsm8k_grpo_qwen7b",
    num_train_epochs = 1,               # one epoch is usually enough for a reasoning boost
    per_device_train_batch_size = 2,    # bump this up to 4-8 if vram allows
    gradient_accumulation_steps = 8,    # effective batch ~16
    learning_rate = 5e-6,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    logging_steps = 5,
    save_strategy = "steps",
    save_steps = 200,
    max_steps = -1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    report_to = "none",                 # switch to "wandb" if you want logging

    # grpo specific - keep generations small for speed
    num_generations = 8,                # 4-16; smaller = faster
    group_size = 4,
    max_length = max_seq_length,
    max_prompt_length = 512,
    generation_kwargs = dict(
        max_new_tokens = 384,
        temperature = 0.7,
        top_p = 0.95,
        do_sample = True,
    ),
    use_vllm = False,                   # set True if vllm is installed for 2-4x faster generation
    # vllm_gpu_memory_utilization = 0.75,
)

trainer = GRPOTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    reward_funcs = reward_funcs,
    # packing = False,          # usually False for grpo
)

trainer.train()

# save the final model
trainer.save_model("gsm8k_grpo_qwen7b_final")
# model.save_pretrained_merged("gsm8k_grpo_merged", tokenizer, save_method = "merged_16bit")