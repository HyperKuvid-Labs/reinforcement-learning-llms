from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .aime import build_prompt, compute_binary_reward
from .algorithms import LossBundle, compute_group_advantages, dppo_loss, ppo_loss
from .auth import ensure_hf_credentials
from .config import RunConfig
from .divergence import (
    binary_kl,
    binary_tv,
    current_reduced_distribution,
    full_distribution_kl,
    full_distribution_tv,
    old_reduced_distribution,
    topk_kl,
    topk_tv,
)
from .tb import TensorBoardLogger


class StatusSink:
    def __init__(self, status_file: Path | None) -> None:
        self.status_file = status_file

    def write(self, payload: dict[str, object]) -> None:
        if not self.status_file:
            return
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _import_stack():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    return (
        torch,
        load_dataset,
        HfApi,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


class Trainer:
    def __init__(self, config: RunConfig, callback=None) -> None:
        self.config = config
        self.callback = callback
        self.status_sink = StatusSink(config.status_file)
        self.tb = TensorBoardLogger(config.run_dir)
        self.global_step = 0
        self.last_status: dict[str, object] = {}

    def _emit(self, phase: str, **metrics) -> None:
        payload = {
            "phase": phase,
            "global_step": self.global_step,
            "run_name": self.config.slug,
            "model_id": self.config.model_id,
            "algo": self.config.algo,
            "dppo_approx": self.config.dppo_approx,
            **metrics,
        }
        self.last_status = payload
        self.status_sink.write(payload)
        if self.callback:
            self.callback(payload)

    @property
    def _trainer_state_path(self) -> Path:
        return self.config.checkpoint_dir / "trainer_state.pt"

    def _offload_dir(self) -> Path:
        path = self.config.offload_root / self.config.slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _normalize_finetune_method(self) -> None:
        if self.config.finetune_method == "qlora" and "HRM-Text-1B" in self.config.model_id:
            self._emit(
                "adapter_fallback",
                requested_method="qlora",
                fallback_method="lora",
                reason="hrm remote-code architecture is not stable with the generic qlora injection path",
            )
            self.config.finetune_method = "lora"

    def _build_model_load_kwargs(self, torch, BitsAndBytesConfig):
        kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        if self.config.finetune_method == "qlora":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        if self.config.cpu_offload:
            kwargs["device_map"] = "auto"
            kwargs["offload_folder"] = str(self._offload_dir())
            kwargs["offload_state_dict"] = True
        elif torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        return kwargs

    @staticmethod
    def _lora_target_modules():
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def _apply_adapter_training(self, model, LoraConfig, get_peft_model, prepare_model_for_kbit_training):
        if self.config.finetune_method == "full":
            return model
        requested_method = self.config.finetune_method
        try:
            if requested_method == "qlora":
                model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self._lora_target_modules(),
            )
            model = get_peft_model(model, peft_config)
            return model
        except Exception as exc:
            if requested_method != "qlora":
                raise
            self._emit(
                "adapter_fallback",
                requested_method=requested_method,
                fallback_method="lora",
                reason=str(exc),
            )
            self.config.finetune_method = "lora"
            if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
                raise RuntimeError(
                    "qlora injection failed for this model after 4-bit load. rerun with --finetune-method lora."
                ) from exc
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self._lora_target_modules(),
            )
            return get_peft_model(model, peft_config)

    def _load_components(self):
        self._normalize_finetune_method()
        (
            torch,
            load_dataset,
            HfApi,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        ) = _import_stack()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **self._build_model_load_kwargs(torch, BitsAndBytesConfig),
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model = self._apply_adapter_training(
            model,
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        if not self.config.cpu_offload and torch.cuda.is_available():
            model = model.to("cuda")
        dataset = load_dataset(self.config.dataset_id, split=self.config.dataset_split)
        if self.config.train_examples_limit:
            dataset = dataset.select(range(min(len(dataset), self.config.train_examples_limit)))
        return torch, tokenizer, model, dataset, HfApi

    def _prepare_hf_auth(self) -> tuple[str, str]:
        self._emit("auth")
        username, token = ensure_hf_credentials(prompt=True)
        self.tb.add_text("run/hf_username", username, step=0)
        return username, token

    @staticmethod
    def _device_of(model):
        return next(model.parameters()).device

    def _trainable_parameters(self, model):
        return [param for param in model.parameters() if param.requires_grad]

    def _prepare_batch(self, torch, tokenizer, examples):
        prompts = [build_prompt(example["question"]) for example in examples]
        answers = [str(example["answer"]) for example in examples]
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_tokens,
        )
        model_inputs["pad_token_id"] = tokenizer.pad_token_id
        return prompts, answers, model_inputs

    def _generate_rollouts(self, torch, model, tokenizer, model_inputs, answers):
        batch_size = model_inputs["input_ids"].shape[0]
        device = self._device_of(model)
        repeated = {
            key: value.repeat_interleave(self.config.rollout_group_size, dim=0).to(device)
            for key, value in model_inputs.items()
            if key != "pad_token_id"
        }
        generation = model.generate(
            **repeated,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        sequences = generation.sequences
        prompt_len = repeated["input_ids"].shape[1]
        completions = sequences[:, prompt_len:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        expanded_answers = []
        for answer in answers:
            expanded_answers.extend([answer] * self.config.rollout_group_size)
        rewards = torch.tensor(
            [compute_binary_reward(text, answer) for text, answer in zip(decoded, expanded_answers)],
            device=device,
            dtype=torch.float32,
        ).view(batch_size, self.config.rollout_group_size)

        old_logprobs = []
        old_chosen_probs = []
        old_topk_indices = []
        old_topk_probs = []
        for step_scores, sampled_tokens in zip(generation.scores, completions.transpose(0, 1)):
            probs = step_scores.softmax(dim=-1)
            chosen_probs = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            topk_probs, topk_indices = probs.topk(k=min(self.config.dppo_topk, probs.shape[-1]), dim=-1)
            old_logprobs.append(chosen_probs.clamp_min(1e-12).log())
            old_chosen_probs.append(chosen_probs)
            old_topk_indices.append(topk_indices)
            old_topk_probs.append(topk_probs)

        rollout = {
            "prompt_len": prompt_len,
            "pad_token_id": model_inputs["pad_token_id"],
            "sequences": sequences,
            "completions": completions,
            "decoded": decoded,
            "rewards": rewards,
            "old_logprobs": torch.stack(old_logprobs, dim=1),
            "old_chosen_probs": torch.stack(old_chosen_probs, dim=1),
            "old_topk_indices": torch.stack(old_topk_indices, dim=1),
            "old_topk_probs": torch.stack(old_topk_probs, dim=1),
        }
        return rollout

    def _current_policy_stats(self, torch, model, rollout):
        device = self._device_of(model)
        sequences = rollout["sequences"].to(device)
        prompt_len = rollout["prompt_len"]
        attention_mask = (sequences[:, :-1] != rollout["pad_token_id"]).long()
        outputs = model(input_ids=sequences[:, :-1], attention_mask=attention_mask)
        logits = outputs.logits[:, prompt_len - 1 :, :]
        target_tokens = sequences[:, prompt_len:]
        current_logprobs = logits.log_softmax(dim=-1).gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
        current_probs = logits.softmax(dim=-1)
        chosen_probs = current_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

        reduced_old = []
        reduced_current = []
        for step in range(target_tokens.shape[1]):
            reduced_old.append(
                old_reduced_distribution(
                    rollout["old_topk_indices"][:, step, :],
                    rollout["old_topk_probs"][:, step, :],
                    target_tokens[:, step],
                    rollout["old_chosen_probs"][:, step],
                )
            )
            reduced_current.append(
                current_reduced_distribution(
                    rollout["old_topk_indices"][:, step, :],
                    target_tokens[:, step],
                    current_probs[:, step, :],
                )
            )
        return {
            "current_logprobs": current_logprobs,
            "chosen_probs": chosen_probs,
            "current_probs": current_probs,
            "reduced_old": torch.stack(reduced_old, dim=1),
            "reduced_current": torch.stack(reduced_current, dim=1),
        }

    def _loss_for_algo(self, torch, rollout, stats):
        batch = rollout["rewards"].shape[0]
        steps = stats["current_logprobs"].shape[1]
        advantages = compute_group_advantages(
            rollout["rewards"], "grpo" if self.config.algo == "grpo" else "ppo"
        )
        advantages = advantages.reshape(-1, 1).expand(batch * self.config.rollout_group_size, steps)
        old_logprobs = rollout["old_logprobs"]
        current_logprobs = stats["current_logprobs"]

        if self.config.algo == "grpo":
            loss = -(current_logprobs * advantages).mean()
            return LossBundle(loss=loss, metrics={})
        if self.config.algo == "ppo":
            return ppo_loss(current_logprobs, old_logprobs, advantages, self.config.ppo_clip_eps)
        return dppo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            current_chosen_probs=stats["chosen_probs"],
            old_chosen_probs=rollout["old_chosen_probs"],
            current_topk_probs=stats["reduced_current"],
            old_topk_probs=stats["reduced_old"],
            approx=self.config.dppo_approx,
            delta=self.config.dppo_delta,
        )

    def _save_trainer_state(self, torch, model, optimizer) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model": (model.module if hasattr(model, "module") else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(state, self._trainer_state_path)

    def _checkpoint(self, torch, model, tokenizer, optimizer=None) -> dict[str, float]:
        start = perf_counter()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        if optimizer is not None:
            self._save_trainer_state(torch, model, optimizer)
        model_to_save.save_pretrained(self.config.checkpoint_dir)
        tokenizer.save_pretrained(self.config.checkpoint_dir)
        upload_seconds = 0.0
        prune_status = 0.0
        if self.config.push_to_hub and self.config.hub_repo:
            self._emit("uploading", checkpoint_dir=str(self.config.checkpoint_dir))
            _, _, _, _, _, HfApi = _import_stack()
            api = HfApi()
            api.upload_folder(
                repo_id=self.config.hub_repo,
                folder_path=str(self.config.checkpoint_dir),
                path_in_repo=self.config.slug,
                token=self.hf_token,
            )
            upload_seconds = perf_counter() - start
            if self.config.delete_local_checkpoints:
                shutil.rmtree(self.config.checkpoint_dir, ignore_errors=True)
                prune_status = 1.0
        return {
            "system/checkpoint_upload_time": upload_seconds,
            "system/checkpoint_prune_status": prune_status,
        }

    def train(self) -> dict[str, object]:
        self.hf_username, self.hf_token = self._prepare_hf_auth()
        if self.config.push_to_hub and not self.config.hub_repo:
            self.config.hub_repo = self.config.default_hub_repo(self.hf_username)
        torch, tokenizer, model, dataset, _ = self._load_components()
        optimizer = torch.optim.AdamW(
            self._trainable_parameters(model),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        if self.config.resume == "auto" and self._trainer_state_path.exists():
            checkpoint = torch.load(self._trainer_state_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.global_step = int(checkpoint.get("global_step", 0))
        elif self.config.resume not in {"off", "auto"}:
            checkpoint = torch.load(self.config.resume, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.global_step = int(checkpoint.get("global_step", 0))

        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb.add_hparams_blob(asdict(self.config))
        self._emit("ready", run_dir=str(self.config.run_dir))

        for example in dataset:
            self.global_step += 1
            _, answers, model_inputs = self._prepare_batch(torch, tokenizer, [example])
            rollout_start = perf_counter()
            self._emit("sampling", prompt=example["question"][:120])
            rollout = self._generate_rollouts(torch, model, tokenizer, model_inputs, answers)
            reward_mean = float(rollout["rewards"].mean().item())
            reward_std = float(rollout["rewards"].std(unbiased=False).item())

            step_metrics: dict[str, float] = {
                "train/reward_mean": reward_mean,
                "train/reward_std": reward_std,
                "eval/reward_mean": reward_mean,
            }
            for epoch_idx in range(self.config.ppo_epochs):
                self._emit("backward", ppo_epoch=epoch_idx + 1)
                stats = self._current_policy_stats(torch, model, rollout)
                bundle = self._loss_for_algo(torch, rollout, stats)
                optimizer.zero_grad()
                bundle.loss.backward()
                optimizer.step()

                adv = compute_group_advantages(
                    rollout["rewards"], "grpo" if self.config.algo == "grpo" else "ppo"
                )
                step_metrics.update(
                    {
                        "train/loss": float(bundle.loss.item()),
                        "train/advantage_mean": float(adv.mean().item()),
                        "train/advantage_std": float(adv.std(unbiased=False).item()),
                        "eval/accuracy": reward_mean,
                        "eval/completion_length": float(rollout["completions"].shape[1]),
                        "system/step_time": perf_counter() - rollout_start,
                        **bundle.metrics,
                    }
                )
                if torch.cuda.is_available():
                    step_metrics["system/gpu_mem_allocated"] = float(torch.cuda.memory_allocated())
                    step_metrics["system/gpu_mem_reserved"] = float(torch.cuda.memory_reserved())

                tokens_per_sec = 0.0
                elapsed = max(perf_counter() - rollout_start, 1e-6)
                tokens_per_sec = float(rollout["completions"].numel() / elapsed)
                step_metrics["system/tokens_per_sec"] = tokens_per_sec

                self.tb.add_scalars(self.global_step, step_metrics)
                self.tb.flush()

            if self.global_step % self.config.save_every == 0:
                self._emit("saving")
                checkpoint_metrics = self._checkpoint(torch, model, tokenizer, optimizer)
                self.tb.add_scalars(self.global_step, checkpoint_metrics)

            if self.global_step % self.config.log_every == 0:
                self._emit("running", **step_metrics)

        self._emit("done", total_steps=self.global_step)
        self.tb.close()
        return self.last_status

    def compare_divergence(self) -> dict[str, float]:
        self.hf_username, self.hf_token = self._prepare_hf_auth()
        torch, tokenizer, model, dataset, _ = self._load_components()
        optimizer = torch.optim.AdamW(self._trainable_parameters(model), lr=self.config.learning_rate)
        example = dataset[0]
        _, answers, model_inputs = self._prepare_batch(torch, tokenizer, [example])
        rollout = self._generate_rollouts(torch, model, tokenizer, model_inputs, answers)
        before = self._current_policy_stats(torch, model, rollout)
        bundle = self._loss_for_algo(torch, rollout, before)
        optimizer.zero_grad()
        bundle.loss.backward()
        optimizer.step()
        stats = self._current_policy_stats(torch, model, rollout)

        flat_mu = rollout["old_chosen_probs"].reshape(-1)
        flat_pi = stats["chosen_probs"].reshape(-1)
        binary_tv_mean = float(binary_tv(flat_mu, flat_pi).mean().item())
        binary_kl_mean = float(binary_kl(flat_mu, flat_pi).mean().item())

        reduced_old = stats["reduced_old"].reshape(-1, stats["reduced_old"].shape[-1])
        reduced_current = stats["reduced_current"].reshape(-1, stats["reduced_current"].shape[-1])
        topk_tv_mean = float(topk_tv(reduced_old, reduced_current).mean().item())
        topk_kl_mean = float(topk_kl(reduced_old, reduced_current).mean().item())

        current_probs = stats["current_probs"].reshape(-1, stats["current_probs"].shape[-1])
        prompt_len = rollout["prompt_len"]
        sequences = rollout["sequences"].to(self._device_of(model))
        attention_mask = (sequences[:, :-1] != rollout["pad_token_id"]).long()
        outputs = model(input_ids=sequences[:, :-1], attention_mask=attention_mask)
        full_pi = outputs.logits[:, prompt_len - 1 :, :].softmax(dim=-1).reshape(-1, outputs.logits.shape[-1])
        flat_indices = rollout["completions"].reshape(-1)
        old_chosen = rollout["old_chosen_probs"].reshape(-1)
        full_mu = full_pi.clone()
        replacement = full_mu.gather(1, flat_indices.unsqueeze(-1)).squeeze(-1)
        remainder_scale = ((1.0 - old_chosen).clamp_min(1e-12) / (1.0 - replacement).clamp_min(1e-12)).unsqueeze(-1)
        full_mu = full_mu * remainder_scale
        full_mu.scatter_(1, flat_indices.unsqueeze(-1), old_chosen.unsqueeze(-1))
        naive_tv_mean = float(full_distribution_tv(full_mu, full_pi).mean().item())
        naive_kl_mean = float(full_distribution_kl(full_mu, full_pi).mean().item())

        metrics = {
            "divergence/binary_tv": binary_tv_mean,
            "divergence/binary_kl": binary_kl_mean,
            "divergence/topk_tv": topk_tv_mean,
            "divergence/topk_kl": topk_kl_mean,
            "divergence/naive_tv": naive_tv_mean,
            "divergence/naive_kl": naive_kl_mean,
        }
        self.tb.add_scalars(0, metrics)
        self.tb.close()
        self._emit("done", **metrics)
        return metrics
