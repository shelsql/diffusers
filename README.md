[Original README by Huggingface here](README_huggingface.md)

# Wan2.2 diffusers workspace
Sheldon Liang, Yinghao Zhang

## Task 1: Integrate Flash Attention 3 into diffusers Wan2.2

### Environment

```bash
./env.sh
```

### Testing

```bash
python test.py
```
When running for the first time, the script will first download models from huggingface to cache, takes ~10 minutes.