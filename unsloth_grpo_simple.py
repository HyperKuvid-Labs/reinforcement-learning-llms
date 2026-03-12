from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import re
from typing import Optional

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 512
dtype = None  # ← explicit dtype instead of None
load_in_4bit = True
model_name = "Qwen/Qwen3-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=False,  # ← set to False to avoid conflict with load_in_16bit
    load_in_16bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Fix for trl/peft compatibility: GRPOTrainer expects warnings_issued attribute
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

dataset = load_dataset("openai/gsm8k", "main", split="train")


SYSTEM_PROMPT = """\
<system>
  <role>You are a mathematical reasoning assistant.</role>
  <instructions>
    <step>Think through the problem carefully inside &lt;think&gt;...&lt;/think&gt; tags.</step>
    <step>Show your full step-by-step reasoning inside the think block.</step>
    <step>Provide your final numeric answer inside \\boxed{{...}}.</step>
  </instructions>
  <format>
    <think>step-by-step reasoning here</think>
    \\boxed{{final answer}}
  </format>
</system>"""


def formatting_prompts_func(examples):
    texts = [
        f"{SYSTEM_PROMPT}\n\nQuestion: {q}" for q in examples["question"]
    ]
    return {
        "prompt": texts,
        "ground_truth": [a.split("####")[-1].strip() for a in examples["answer"]],
    }


dataset = dataset.map(formatting_prompts_func, batched=True)


def extract_boxed_answer(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{(.*?)\}", text)
    return m.group(1).strip() if m else None


def accuracy_reward_func(completions, ground_truth, **kwargs):
    return [
        1.0 if extract_boxed_answer(c) == gt else 0.0
        for c, gt in zip(completions, ground_truth)
    ]


def format_reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        has_think = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        has_boxed = bool(re.search(r"\\\\boxed\{.*?\}", c, re.DOTALL))
        if has_think and has_boxed:
            rewards.append(1.0)
        elif has_boxed:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


reward_funcs = [accuracy_reward_func, format_reward_func]

training_args = GRPOConfig(
    output_dir="outputs/gsm8k_grpo_qwen4b",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="steps",
    save_steps=200,
    max_steps=-1,
    report_to="tensorboard",
    num_generations=8,
    max_prompt_length=256,
    # max_completion_length=192, commenting this out to basically, the math problems need more tokens to think and solve
    temperature=0.7,
    top_p=0.95,
    use_vllm=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_funcs,
    # tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("gsm8k_grpo_qwen3b_final_unlsoth_trl")
