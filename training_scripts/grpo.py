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
    # output / logging
    output_dir="qwen3-grpo-gsm8k",
    run_name="qwen3-grpo-A100",
    logging_steps=5,              # logging every step slows things a bit, this is cleaner

    # hyperparams
    learning_rate=1e-5,           # 0.6b is small, can push lr slightly higher without exploding
    beta=0.04,                    # standard kl region, keeps updates stable
    num_iterations=1,             # rl loops usually run 1 pass per batch

    # generation (actual grpo part)
    num_generations=8,            # 16 sounds nice but rm becomes bottleneck on single a100
    max_completion_length=512,    # 1024 wastes kv cache unless answers are super long
    temperature=0.8,              # encourages diverse reasoning paths
    top_p=0.95,
    top_k=50,

    # batch sizing tuned for a100 80gb
    per_device_train_batch_size=32,   # model is tiny, no reason to stay low
    generation_batch_size=96,         # main throughput lever for vllm
    gradient_accumulation_steps=1,    # fits natively, no need to accumulate

    # vllm acceleration
    use_vllm=True,
    vllm_mode="colocate",             # fastest for single gpu setup
    vllm_gpu_memory_utilization=0.75, # 0.6 underuses a100 memory

    # optimization
    bf16=True,                        # a100 loves bf16
    gradient_checkpointing=False,     # saves compute since model is small

    # grpo specifics
    epsilon=0.2,                      # slightly wider clip is more stable here
    loss_type="grpo",
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