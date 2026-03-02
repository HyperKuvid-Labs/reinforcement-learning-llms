from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import re

MODEL_NAME = "Qwen/Qwen3.5-2B"  

system_prompt = "You are an expert mathematician. Think step by step and put the final answer in \\boxed{}."

def normalize_answer(text):
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_boxed(s):
    match = re.search(r'\\boxed\{(.*)\}', s, re.DOTALL)
    return match.group(1).strip() if match else ""

def test():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    correct = 0

    for i, example in enumerate(dataset):
        question = example["question"]
        true_answer = normalize_answer(example["answer"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False
            )

        response = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        pred = extract_boxed(response)
        pred_norm = normalize_answer(pred)

        if pred_norm == true_answer:
            correct += 1

        print(f"{i+1:3d} {'✓' if pred_norm == true_answer else '✗'} | {pred_norm} | {true_answer}")

    print(f"Accuracy: {correct / len(dataset):.3%} ({correct}/{len(dataset)})")

if __name__ == "__main__":
    test()