from __future__ import annotations

import json
import os
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .aime import build_prompt, compute_binary_reward, extract_final_answer, normalize_answer
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


def _import_unsloth_stack():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth backend requested but `unsloth` is not installed. Run `bash install.sh` or install `unsloth`."
        ) from exc

    return torch, load_dataset, HfApi, FastLanguageModel


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

    def _resolved_backend(self) -> str:
        if self.config.trainer_backend != "auto":
            return self.config.trainer_backend
        return "unsloth" if self.config.model_id.startswith("Qwen/") else "transformers"

    def _build_model_load_kwargs(self, torch, BitsAndBytesConfig):
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
            "low_cpu_mem_usage": True,
        }
        if self.config.finetune_method == "qlora":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
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
            "gqkv_proj",
            "gate_up_proj",
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
        if self._resolved_backend() == "unsloth":
            return self._load_components_unsloth()
        return self._load_components_transformers()

    def _load_components_unsloth(self):
        self._emit(
            "load_start",
            trainer_backend=self._resolved_backend(),
            finetune_method=self.config.finetune_method,
            cpu_offload=self.config.cpu_offload,
            dataset_split=self.config.dataset_split,
        )
        torch, load_dataset, HfApi, FastLanguageModel = _import_unsloth_stack()
        if self.config.cpu_offload:
            self._emit(
                "hparam_adjustment",
                field="cpu_offload",
                old_value=True,
                new_value=False,
                reason="unsloth manages placement internally; transformers device_map offload is not used",
            )
            self.config.cpu_offload = False
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        load_in_4bit = self.config.finetune_method == "qlora"
        load_kwargs = {
            "model_name": self.config.model_id,
            "max_seq_length": self.config.max_prompt_tokens + self.config.max_new_tokens,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "fast_inference": False,
            "token": getattr(self, "hf_token", None),
        }
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        except TypeError:
            load_kwargs.pop("token", None)
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
            except TypeError:
                load_kwargs.pop("fast_inference", None)
                model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        self._emit("tokenizer_loaded")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.use_cache = False
        self._emit(
            "model_loaded",
            cuda_available=bool(torch.cuda.is_available()),
            model_device=str(self._device_of(model)),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        )
        if self.config.finetune_method in {"lora", "qlora"}:
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self._lora_target_modules(),
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
            )
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(FastLanguageModel, "for_training"):
            model = FastLanguageModel.for_training(model)
        self._emit(
            "adapter_ready",
            finetune_method=self.config.finetune_method,
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        self._emit(
            "dataset_loading",
            dataset=self.config.dataset_id,
            dataset_config=self.config.dataset_config,
            split=self.config.dataset_split,
        )
        dataset = self._load_dataset(load_dataset)
        self._emit("dataset_loaded", examples=len(dataset))
        return torch, tokenizer, model, dataset, HfApi

    def _load_components_transformers(self):
        self._normalize_finetune_method()
        self._emit(
            "load_start",
            trainer_backend=self._resolved_backend(),
            finetune_method=self.config.finetune_method,
            cpu_offload=self.config.cpu_offload,
            dataset_split=self.config.dataset_split,
        )
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
        self._emit("tokenizer_loaded")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **self._build_model_load_kwargs(torch, BitsAndBytesConfig),
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        self._emit(
            "model_loaded",
            cuda_available=bool(torch.cuda.is_available()),
            model_device=str(self._device_of(model)),
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
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
        self._emit(
            "adapter_ready",
            finetune_method=self.config.finetune_method,
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        self._emit(
            "dataset_loading",
            dataset=self.config.dataset_id,
            dataset_config=self.config.dataset_config,
            split=self.config.dataset_split,
        )
        dataset = self._load_dataset(load_dataset)
        self._emit("dataset_loaded", examples=len(dataset))
        return torch, tokenizer, model, dataset, HfApi

    def _load_dataset(self, load_dataset):
        if self.config.dataset_config:
            dataset = load_dataset(
                self.config.dataset_id,
                self.config.dataset_config,
                split=self.config.dataset_split,
            )
        else:
            dataset = load_dataset(self.config.dataset_id, split=self.config.dataset_split)
        if self.config.train_examples_limit:
            dataset = dataset.select(range(min(len(dataset), self.config.train_examples_limit)))
        return dataset

    def _prepare_hf_auth(self) -> tuple[str, str]:
        self._emit("auth")
        username, token = ensure_hf_credentials(prompt=True)
        self.tb.add_text("run/hf_username", username, step=0)
        return username, token

    def _normalize_full_training_hparams(self) -> None:
        if self.config.finetune_method == "full" and self.config.learning_rate >= 5e-6:
            self._emit(
                "hparam_adjustment",
                field="learning_rate",
                old_value=self.config.learning_rate,
                new_value=1e-6,
                reason="full finetune is numerically unstable at the higher default lr",
            )
            self.config.learning_rate = 1e-6

    @staticmethod
    def _device_of(model):
        return next(model.parameters()).device

    def _trainable_parameters(self, model):
        return [param for param in model.parameters() if param.requires_grad]

    @staticmethod
    def _text_tokenizer(tokenizer):
        return getattr(tokenizer, "tokenizer", tokenizer)

    def _assert_finite_tensor(self, torch, tensor, label: str) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite tensor detected in {label}")

    def _assert_finite_trainable_params(self, torch, model) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all():
                raise RuntimeError(f"non-finite parameter detected after optimizer step: {name}")

    def _prepare_batch(self, torch, tokenizer, examples):
        prompts = [build_prompt(example["question"]) for example in examples]
        answers = [str(example["answer"]) for example in examples]
        text_tokenizer = self._text_tokenizer(tokenizer)
        if text_tokenizer is not tokenizer:
            model_inputs = text_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
            )
        else:
            model_inputs = tokenizer(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
            )
        model_inputs["pad_token_id"] = text_tokenizer.pad_token_id
        return prompts, answers, model_inputs

    def _generate_rollouts(self, torch, model, tokenizer, model_inputs, answers):
        batch_size = model_inputs["input_ids"].shape[0]
        device = self._device_of(model)
        needs_old_scores = self.config.algo in {"ppo", "dppo"}
        needs_dppo_stats = self.config.algo == "dppo"
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
            output_scores=needs_old_scores,
            pad_token_id=tokenizer.pad_token_id,
        )
        sequences = generation.sequences
        prompt_len = repeated["input_ids"].shape[1]
        completions = sequences[:, prompt_len:]
        decoded = self._text_tokenizer(tokenizer).batch_decode(completions, skip_special_tokens=True)
        expanded_answers = []
        for answer in answers:
            expanded_answers.extend([answer] * self.config.rollout_group_size)
        extracted_answers = [extract_final_answer(text) for text in decoded]
        normalized_predictions = [normalize_answer(answer) for answer in extracted_answers]
        normalized_answers = [normalize_answer(extract_final_answer(answer)) for answer in expanded_answers]
        rewards = torch.tensor(
            [compute_binary_reward(text, answer) for text, answer in zip(decoded, expanded_answers)],
            device=device,
            dtype=torch.float32,
        ).view(batch_size, self.config.rollout_group_size)

        old_logprobs = []
        old_chosen_probs = []
        old_topk_indices = []
        old_topk_probs = []
        if needs_old_scores:
            for step_scores, sampled_tokens in zip(generation.scores, completions.transpose(0, 1)):
                probs = step_scores.softmax(dim=-1)
                chosen_probs = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                old_logprobs.append(chosen_probs.clamp_min(1e-12).log())
                if needs_dppo_stats:
                    topk_probs, topk_indices = probs.topk(k=min(self.config.dppo_topk, probs.shape[-1]), dim=-1)
                    old_chosen_probs.append(chosen_probs)
                    old_topk_indices.append(topk_indices)
                    old_topk_probs.append(topk_probs)

        rollout = {
            "prompt_len": prompt_len,
            "pad_token_id": model_inputs["pad_token_id"],
            "sequences": sequences,
            "completions": completions,
            "decoded": decoded,
            "extracted_answers": extracted_answers,
            "normalized_predictions": normalized_predictions,
            "normalized_answers": normalized_answers,
            "rewards": rewards,
        }
        if needs_old_scores:
            rollout["old_logprobs"] = torch.stack(old_logprobs, dim=1)
        if needs_dppo_stats:
            rollout["old_chosen_probs"] = torch.stack(old_chosen_probs, dim=1)
            rollout["old_topk_indices"] = torch.stack(old_topk_indices, dim=1)
            rollout["old_topk_probs"] = torch.stack(old_topk_probs, dim=1)
        del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rollout

    def _current_policy_stats(self, torch, model, rollout, row_slice=None, include_full_probs: bool = False):
        device = self._device_of(model)
        row_slice = row_slice if row_slice is not None else slice(None)
        sequences = rollout["sequences"][row_slice].to(device)
        prompt_len = rollout["prompt_len"]
        attention_mask = (sequences[:, :-1] != rollout["pad_token_id"]).long()
        outputs = model(input_ids=sequences[:, :-1], attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, prompt_len - 1 :, :]
        target_tokens = sequences[:, prompt_len:]
        current_logprobs = logits.log_softmax(dim=-1).gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
        stats = {"current_logprobs": current_logprobs}
        if self.config.algo == "dppo":
            chosen_probs = current_logprobs.exp()
            stats["chosen_probs"] = chosen_probs
            if self.config.dppo_approx == "topk" or include_full_probs:
                current_probs = logits.softmax(dim=-1)
                if include_full_probs:
                    stats["current_probs"] = current_probs
                if self.config.dppo_approx == "topk":
                    reduced_old = []
                    reduced_current = []
                    for step in range(target_tokens.shape[1]):
                        reduced_old.append(
                            old_reduced_distribution(
                                rollout["old_topk_indices"][row_slice, step, :],
                                rollout["old_topk_probs"][row_slice, step, :],
                                target_tokens[:, step],
                                rollout["old_chosen_probs"][row_slice, step],
                            )
                        )
                        reduced_current.append(
                            current_reduced_distribution(
                                rollout["old_topk_indices"][row_slice, step, :],
                                target_tokens[:, step],
                                current_probs[:, step, :],
                            )
                        )
                    stats["reduced_old"] = torch.stack(reduced_old, dim=1)
                    stats["reduced_current"] = torch.stack(reduced_current, dim=1)
        return stats

    def _loss_for_algo(self, torch, rollout, stats, row_slice=None, advantages=None):
        row_slice = row_slice if row_slice is not None else slice(None)
        total_rows = rollout["sequences"].shape[0]
        steps = stats["current_logprobs"].shape[1]
        if advantages is None:
            advantages = compute_group_advantages(
                rollout["rewards"], "grpo" if self.config.algo == "grpo" else "ppo"
            )
            advantages = advantages.reshape(total_rows, 1).expand(total_rows, steps)
        advantages = advantages[row_slice]
        current_logprobs = stats["current_logprobs"]

        if self.config.algo == "grpo":
            loss = -(current_logprobs * advantages).mean()
            return LossBundle(loss=loss, metrics={})
        old_logprobs = rollout["old_logprobs"][row_slice]
        if self.config.algo == "ppo":
            return ppo_loss(current_logprobs, old_logprobs, advantages, self.config.ppo_clip_eps)
        return dppo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            current_chosen_probs=stats["chosen_probs"],
            old_chosen_probs=rollout["old_chosen_probs"][row_slice],
            current_topk_probs=stats.get("reduced_current"),
            old_topk_probs=stats.get("reduced_old"),
            approx=self.config.dppo_approx,
            delta=self.config.dppo_delta,
        )

    def _policy_update_epoch(self, torch, model, optimizer, rollout) -> dict[str, float]:
        total_rows = int(rollout["sequences"].shape[0])
        steps = int(rollout["completions"].shape[1])
        micro_batch_size = max(1, int(self.config.micro_batch_size))
        micro_batches = (total_rows + micro_batch_size - 1) // micro_batch_size
        mode = "grpo" if self.config.algo == "grpo" else "ppo"
        advantages = compute_group_advantages(rollout["rewards"], mode).reshape(total_rows, 1).expand(total_rows, steps)

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        metric_totals: dict[str, float] = {}
        for start in range(0, total_rows, micro_batch_size):
            end = min(total_rows, start + micro_batch_size)
            row_slice = slice(start, end)
            weight = (end - start) / total_rows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stats = self._current_policy_stats(torch, model, rollout, row_slice=row_slice)
            bundle = self._loss_for_algo(torch, rollout, stats, row_slice=row_slice, advantages=advantages)
            self._assert_finite_tensor(torch, bundle.loss.detach(), "loss")
            (bundle.loss * weight).backward()
            loss_total += float(bundle.loss.detach().item()) * weight
            for key, value in bundle.metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * weight
            del stats, bundle

        self._emit(
            "stats_ready",
            sequence_steps=steps,
            micro_batch_size=micro_batch_size,
            micro_batches=micro_batches,
        )
        trainable = self._trainable_parameters(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm detected before optimizer step")
        optimizer.step()
        self._assert_finite_trainable_params(torch, model)

        metrics = {
            "train/loss": loss_total,
            "train/advantage_mean": float(advantages.mean().item()),
            "train/advantage_std": float(advantages.std(unbiased=False).item()),
            "train/grad_norm": float(grad_norm.item()),
        }
        metrics.update(metric_totals)
        return metrics

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
            from huggingface_hub import HfApi

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
        try:
            self.hf_username, self.hf_token = self._prepare_hf_auth()
            if self.config.push_to_hub and not self.config.hub_repo:
                self.config.hub_repo = self.config.default_hub_repo(self.hf_username)
            self._normalize_full_training_hparams()
            torch, tokenizer, model, dataset, _ = self._load_components()
            optimizer = torch.optim.AdamW(
                self._trainable_parameters(model),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self._emit("optimizer_ready", learning_rate=self.config.learning_rate)
            if self.config.resume == "auto" and self._trainer_state_path.exists():
                checkpoint = torch.load(self._trainer_state_path, map_location="cpu")
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                self.global_step = int(checkpoint.get("global_step", 0))
                self._emit("resumed", resume_path=str(self._trainer_state_path), global_step=self.global_step)
            elif self.config.resume not in {"off", "auto"}:
                checkpoint = torch.load(self.config.resume, map_location="cpu")
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                self.global_step = int(checkpoint.get("global_step", 0))
                self._emit("resumed", resume_path=str(self.config.resume), global_step=self.global_step)

            self.config.run_dir.mkdir(parents=True, exist_ok=True)
            self.tb.add_hparams_blob(asdict(self.config))
            self._emit("ready", run_dir=str(self.config.run_dir))

            for example in dataset:
                self.global_step += 1
                _, answers, model_inputs = self._prepare_batch(torch, tokenizer, [example])
                rollout_start = perf_counter()
                self._emit("sampling", prompt=example["question"][:120], step_index=self.global_step)
                rollout = self._generate_rollouts(torch, model, tokenizer, model_inputs, answers)
                self._emit(
                    "rollout_ready",
                    completion_tokens=int(rollout["completions"].numel()),
                    group_size=self.config.rollout_group_size,
                )
                if self.config.reward_debug_every and self.global_step % self.config.reward_debug_every == 0:
                    preview = " ".join(str(rollout["decoded"][0]).split())[:240]
                    self._emit(
                        "sample_debug",
                        reward=float(rollout["rewards"].reshape(-1)[0].item()),
                        pred_answer=rollout["normalized_predictions"][0],
                        gold_answer=rollout["normalized_answers"][0],
                        completion_preview=preview,
                    )
                reward_mean = float(rollout["rewards"].mean().item())
                reward_std = float(rollout["rewards"].std(unbiased=False).item())
                reward_nonzero = float((rollout["rewards"] > 0).float().mean().item())

                step_metrics: dict[str, float] = {
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "train/reward_nonzero_fraction": reward_nonzero,
                    "eval/reward_mean": reward_mean,
                }
                for epoch_idx in range(self.config.ppo_epochs):
                    self._emit("backward", ppo_epoch=epoch_idx + 1)
                    epoch_metrics = self._policy_update_epoch(torch, model, optimizer, rollout)
                    step_metrics.update(
                        {
                            "eval/accuracy": reward_mean,
                            "eval/completion_length": float(rollout["completions"].shape[1]),
                            "system/step_time": perf_counter() - rollout_start,
                            **epoch_metrics,
                        }
                    )
                    if torch.cuda.is_available():
                        step_metrics["system/gpu_mem_allocated"] = float(torch.cuda.memory_allocated())
                        step_metrics["system/gpu_mem_reserved"] = float(torch.cuda.memory_reserved())

                    elapsed = max(perf_counter() - rollout_start, 1e-6)
                    step_metrics["system/tokens_per_sec"] = float(rollout["completions"].numel() / elapsed)

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
        except Exception as exc:
            self._emit("error", error=str(exc), traceback=traceback.format_exc())
            self.tb.close()
            raise

    def compare_divergence(self) -> dict[str, float]:
        self.hf_username, self.hf_token = self._prepare_hf_auth()
        torch, tokenizer, model, dataset, _ = self._load_components()
        optimizer = torch.optim.AdamW(self._trainable_parameters(model), lr=self.config.learning_rate)
        example = dataset[0]
        _, answers, model_inputs = self._prepare_batch(torch, tokenizer, [example])
        rollout = self._generate_rollouts(torch, model, tokenizer, model_inputs, answers)
        before = self._current_policy_stats(torch, model, rollout, include_full_probs=True)
        bundle = self._loss_for_algo(torch, rollout, before)
        optimizer.zero_grad()
        bundle.loss.backward()
        optimizer.step()
        stats = self._current_policy_stats(torch, model, rollout, include_full_probs=True)

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
