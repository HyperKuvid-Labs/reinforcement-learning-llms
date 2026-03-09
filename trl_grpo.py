import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig, setup_vllm
from trl.rewards import reasoning_accuracy_reward

# enabling tf32 for faster matmul on a100 - this helps a lot
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# loading up the dataset and preprocessing it
ds = load_dataset("gsm8k", "main", split="train")  # openai/gsm8k is aliased to gsm8k so either works

def process(example):
    question = example["question"]
    answer = example["answer"]
    solution = answer.split("#### ")[-1].strip()
    prompt = f"Question: {question}\n\nLet's think step by step. Put the final answer within \\boxed{{{solution}}}."
    return {"prompt": prompt, "solution": solution}

dataset = ds.map(process, num_proc=4)

model_name = "Qwen/Qwen3.5-4B"

# spinning up vllm for faster generation, using colocate mode here
vllm_model, vllm_tokenizer = setup_vllm(
    model_name,
    dtype="bfloat16",
    tensor_parallel_size=1  # single gpu
)

# lora config to keep memory usage in check
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# grpo config, tuned for speed on a100
training_args = GRPOConfig(
    output_dir="./grpo_gsm8k",
    num_train_epochs=1,  # one epoch is enough for now
    per_device_train_batch_size=4,  # tweak this depending on your memory
    gradient_accumulation_steps=4,  # effective batch size of 16
    learning_rate=5e-6,
    optim="adamw_torch",
    bf16=True,  # bf16 is the way to go on a100
    tf32=True,
    max_length=512,  # keeping it short to stay fast
    max_prompt_length=256,
    num_generations=8,  # using 8 to keep things moving, paper uses 16+
    group_size=4,  # smaller groups = faster
    generation_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    },
    logging_steps=10,
    save_steps=500,
    use_peft=True,
    peft_config=peft_config,
)

# setting up the trainer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_funcs=reasoning_accuracy_reward,  # reward fn for chain of thought accuracy
    vllm_model=vllm_model,
    vllm_tokenizer=vllm_tokenizer,
)

# let's go
trainer.train()