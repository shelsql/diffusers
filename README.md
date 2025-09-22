[Original README by Huggingface here](README_huggingface.md)

# Wan2.2 diffusers workspace
Sheldon Liang, Yinghao Zhang

## Task 1: Integrate Flash Attention 3 into diffusers Wan2.2

### Environment

```bash
./env.sh
```
Installs all dependencies needed for running Wan2.2 using diffusers, and installs [Flash Attention 3](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release).

\[Untested\] run tests to verify that Flash Attention 3 has been correctly installed:
```bash
cd ../flash-attention/hopper
export PYTHONPATH=$PWD
pytest -q -s test_flash_attn.py
```

### Testing

```bash
python test.py
```
When running for the first time, the script will first download models from huggingface to cache, takes ~10 minutes.

**Test results on 4090 machine:** 1:57 to load checkpoints, 2:03 to run inference.