import numbers
import re
import time
from datetime import timedelta
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainerCallback,
	TrainerControl,
	TrainerState,
)
from trl import GRPOConfig, GRPOTrainer

# enabling tf32 for faster matmul on a100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_name = "Qwen/Qwen3-8B"

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


class MetricsAliasCallback(TrainerCallback):
	def __init__(self) -> None:
		self.tb_writer: Optional[SummaryWriter] = None
		self.accuracy_history: list[tuple[int, float]] = []

	@staticmethod
	def _first_scalar(logs, keys: tuple[str, ...]) -> Optional[float]:
		for key in keys:
			if key in logs and isinstance(logs[key], numbers.Real):
				return float(logs[key])
		return None

	def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
		self.tb_writer = SummaryWriter(log_dir=args.logging_dir)

	def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
		if not logs:
			return

		reward_keys = tuple(k for k in logs if "reward" in k.lower())
		reward_mean = self._first_scalar(
			logs, tuple(k for k in reward_keys if "mean" in k) or reward_keys
		)
		accuracy = self._first_scalar(
			logs,
			(
				"rewards/accuracy_reward_func",
				"accuracy_reward",
				"reward/accuracy",
				"accuracy",
			),
		)
		format_reward = self._first_scalar(
			logs,
			(
				"rewards/format_reward_func",
				"format_reward",
				"reward/format",
			),
		)

		if accuracy is not None:
			self.accuracy_history.append((state.global_step, accuracy))

		aliases: dict[str, Optional[float]] = {
			"loss": self._first_scalar(logs, ("loss", "train_loss")),
			"kl": self._first_scalar(logs, ("kl", "kl_loss", "actor/kl")),
			"lr": self._first_scalar(logs, ("learning_rate", "lr")),
			"grad_norm": self._first_scalar(logs, ("grad_norm", "gradient_norm")),
			"clipped_ratio": self._first_scalar(
				logs,
				(
					"clip_ratio",
					"clipped_ratio",
					"policy/clipped_ratio",
					"actor/clipped_ratio",
				),
			),
			"completion_length": self._first_scalar(
				logs,
				(
					"completion_length",
					"completions/mean_length",
					"mean_completion_length",
				),
			),
			"reward": reward_mean,
			"reward_std": self._first_scalar(
				logs, ("reward_std", "rewards/std", "std_reward")
			),
			"accuracy_r_mean": accuracy,
			"format_r_mean": format_reward,
			"forward_r_mean": format_reward,
		}

		if self.tb_writer is not None:
			for tag, value in aliases.items():
				if value is not None:
					self.tb_writer.add_scalar(tag, value, state.global_step)
			self.tb_writer.flush()

		short_loss = aliases["loss"]
		msg = (
			f"step={state.global_step} "
			f"loss={short_loss:.4f} "
			f"reward={reward_mean:.4f} "
			f"acc={accuracy:.4f} "
			f"fmt={format_reward:.4f}"
		)
		if all(v is not None for v in (short_loss, reward_mean, accuracy, format_reward)):
			print(msg)

	def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
		if self.tb_writer is not None:
			self.tb_writer.close()
			self.tb_writer = None


def process(example):
	ground_truth = example["answer"].split("####")[-1].strip()
	return {
		"prompt": f"{SYSTEM_PROMPT}\n\nQuestion: {example['question']}",
		"ground_truth": ground_truth,
	}


def extract_boxed_answer(text: str) -> Optional[str]:
	match = re.search(r"\\boxed\{(.*?)\}", text)
	return match.group(1).strip() if match else None


def accuracy_reward_func(completions, ground_truth, **kwargs):
	return [
		1.0 if extract_boxed_answer(completion) == gt else 0.0
		for completion, gt in zip(completions, ground_truth)
	]


def format_reward_func(completions, **kwargs):
	rewards = []
	for completion in completions:
		has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
		has_boxed = bool(re.search(r"\\\\boxed\{.*?\}", completion, re.DOTALL))
		if has_think and has_boxed:
			rewards.append(1.0)
		elif has_boxed:
			rewards.append(0.5)
		else:
			rewards.append(0.0)
	return rewards


def main() -> None:
	t0 = time.monotonic()

	print(f"Loading dataset for {model_name}...")
	dataset = load_dataset("gsm8k", "main", split="train")
	dataset = dataset.map(process)
	t_data = time.monotonic()

	print("Loading model/tokenizer...")
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	if not hasattr(model, "warnings_issued"):
		model.warnings_issued = {}

	t_model = time.monotonic()

	training_args = GRPOConfig(
		output_dir="./grpo_gsm8k",
		logging_dir="./grpo_gsm8k/tb_logs",
		num_train_epochs=1,
		per_device_train_batch_size=8,
		gradient_accumulation_steps=4,
		learning_rate=2e-4,
		optim="adamw_8bit",
		weight_decay=0.01,
		warmup_steps=100,
		lr_scheduler_type="cosine",
		bf16=True,
		tf32=True,
		num_generations=8,
		logging_steps=5,
		save_steps=200,
		report_to="tensorboard",
		use_vllm=True,
		vllm_mode="colocate",
		temperature=0.7,
		num_completions_to_print=8,
		top_p=0.95,
	)

	callback = MetricsAliasCallback()
	trainer = GRPOTrainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		reward_funcs=[accuracy_reward_func, format_reward_func],
		callbacks=[callback],
	)

	print("Starting training...")
	train_start = time.monotonic()
	trainer.train()
	train_end = time.monotonic()

	print("\nTraining complete.")
	print(f"- Dataset prep: {timedelta(seconds=int(t_data - t0))}")
	print(f"- Model load:  {timedelta(seconds=int(t_model - t_data))}")
	print(f"- Train loop:  {timedelta(seconds=int(train_end - train_start))}")
	print(f"- Total time:  {timedelta(seconds=int(train_end - t0))}")

	if callback.accuracy_history:
		best_step, best_acc = max(callback.accuracy_history, key=lambda x: x[1])
		last_step, last_acc = callback.accuracy_history[-1]
		print(f"- Best accuracy:  {best_acc * 100:.2f}% (step {best_step})")
		print(f"- Final accuracy: {last_acc * 100:.2f}% (step {last_step})")


if __name__ == "__main__":
	main()
