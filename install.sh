set -e

echo "Starting installation of GRPO / RL libraries stack..."

pip install torch
pip install transformers datasets peft accelerate bitsandbytes -U -q

# TRL (GRPO support)
pip install trl -U -q
pip isntall "trl[vllm]" -U -q || echo "TRL vLLM support install attempted"

# Unsloth (fastest GRPO)
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" -q --no-cache-dir
pip install vllm --upgrade -q || echo "vLLM install skipped (optional)"

# VERL dependencies
pip install hydra-core omegaconf pandas pyarrow sentencepiece protobuf -q
git clone https://github.com/volcengine/verl.git --depth 1 || true
cd verl && pip install -e . -q || echo "VERL editable install attempted"
cd ..

pip install wandb tqdm flash-attn -q || true

echo ""
echo "Key libraries installed:"
python -c "import torch; print('• torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import trl; print('• trl:', trl.__version__)"
python -c "import unsloth; print('• unsloth: OK')"
echo "• verl: installed from source (check folder 'verl')"

echo ""
echo "Installation finished."
echo "You can now run your GRPO training scripts."