import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm_tokenizer = AutoTokenizer.pretrained(reward_model_name)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
).eval()

def skywork_reward_func(prompts, completions, **kwargs):
    inputs = [p + c for p, c in zip(prompts, completions)]

    # Tokenize
    encodings = rm_tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        scores = rm_model(**encodings).logits[:, 0]

    return scores.tolist()

dataset = load_dataset("openai/gsm8k", "main", split="train")

def format_ds(example):
    prompt = (
        "<|im_start|>system\nYou are a helpful math assistant. "
        "Think step by step.<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt}

dataset = dataset.map(format_ds)

config = GRPOConfig(
    output_dir="qwen3-grpo-gsm8k",
    run_name="qwen3-grpo-A100",

    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=1,
    bf16=True,

    num_generations=16,
    max_completion_length=1024,
    beta=0.04,

    per_device_train_batch_size=16,
    generation_batch_size=32,
    gradient_accumulation_steps=1,

    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.5,
)

model_id = "Qwen/Qwen3-0.6B"

trainer = GRPOTrainer(
    model=model_id,
    reward_funcs=[skywork_reward_func],
    train_dataset=dataset,
    args=config,
    processing_class=rm_tokenizer,
)

trainer.train()