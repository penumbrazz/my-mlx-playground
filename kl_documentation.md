# kl.py - KL Divergence Calculator Documentation

## Overview

`kl.py` is a Python script that calculates the Kullback-Leibler (KL) divergence between two language models by comparing their output logits on the same input data. This tool is particularly useful for evaluating the performance difference between a reference model (typically high-precision like fp16) and a quantized model.

## Purpose

The KL divergence measures how one probability distribution diverges from a second, expected probability distribution. In the context of language models, it quantifies how "surprised" a reference model would be if the evaluated model's probability predictions were considered the ground truth.

## Features

- **Model Comparison**: Compare logits between reference and quantized models
- **Efficient Storage**: Compress and store reference logits using gzip
- **Statistical Analysis**: Provides comprehensive statistics including confidence intervals
- **Token-level Analysis**: Calculates top-1, top-5, and top-10 accuracy metrics
- **Flexible Input**: Supports both text files and pre-computed logits files

## Requirements

### Dependencies
```bash
pip install llama-cpp-python numpy scipy
```

### System Requirements
- Python 3.7+
- Sufficient RAM for model loading
- Optional: GPU support for accelerated inference

## Usage

### Basic Workflow

1. **Generate Reference Logits** (using high-precision model):
```bash
python kl.py -m /path/to/fp16/model -t /path/to/text/dataset.txt -w reference_logits.gz
```

2. **Calculate KL Divergence** (using quantized model):
```bash
python kl.py -m /path/to/quantized/model -r reference_logits.gz
```

### Command Line Arguments

| Argument | Short | Description | Required |
|----------|-------|-------------|----------|
| `--model` | `-m` | Path to the model file | Yes |
| `--text_path` | `-t` | Path to text dataset | No* |
| `--n_ctx` | `-c` | Context size (default: 512) | No |
| `--n_batch` | `-b` | Batch size (default: 512) | No |
| `--write_path` | `-w` | Output file for logits | No* |
| `--read_path` | `-r` | Input file with reference logits | No* |
| `--n_tokens` | `-n` | Number of tokens to evaluate (-1 = all) | No |
| `--n_gpu_layers` | `-ngl` | Number of GPU layers (default: 0) | No |
| `--verbose` | `-v` | Enable verbose output | No |

*Either `text_path` or `read_path` must be specified, and at least one of `read_path` or `write_path` must be provided.

## Output Metrics

### KL Divergence Statistics
- **Mean KL**: Average divergence across all tokens
- **Confidence Intervals**: Statistical bounds for the mean
- **Quantiles**: 90th, 95th, and 99th percentiles
- **Maximum**: Highest observed divergence

### Accuracy Metrics
- **Top-1 Agreement**: Percentage of tokens where both models predict the same top token
- **Top-5/Top-10 Overlap**: How often the reference model's top prediction appears in the evaluated model's top-n predictions
- **Reciprocal Overlap**: How often the evaluated model's top prediction appears in the reference model's top-n predictions

## File Formats

### Logits File Structure
The script uses a custom binary format with gzip compression:
1. **Header**: Magic string, configuration parameters
2. **Data Blocks**: For each chunk:
   - Number of tokens (4 bytes)
   - Number of logits (4 bytes)
   - Vocabulary size (4 bytes)
   - Token sequence (4 bytes per token)
   - Logits array (4 bytes per logit value)

### Output Files
- **.kls.p**: Pickle file containing all per-token KL values for further analysis
- **.gz**: Compressed logits file for reuse

## Implementation Details

### Core Functions

#### `kl_div(p, q)`
Calculates KL divergence between two logit vectors:
- Applies softmax to convert logits to probability distributions
- Uses scipy's `rel_entr` for numerical stability
- Returns D_KL(p || q) in nats

#### `write_logits(f, tokens, logits)` / `read_logits(f)`
Handles binary serialization of token sequences and their corresponding logits for efficient storage and retrieval.

### Statistical Methods
- **Bayesian Confidence Intervals**: Uses `scipy.stats.bayes_mvs` for robust interval estimation
- **Quantile CI**: Implements `mquantiles_cimj` for distribution quantiles with confidence bounds
- **Binomial Confidence**: Applies Bayesian smoothing for edge cases in accuracy metrics

## Performance Considerations

### Memory Usage
- Model size + context window loaded into RAM
- Additional memory for batch processing
- Logits storage proportional to tokens processed

### Speed Optimization Tips
- Use appropriate batch sizes for your hardware
- Consider GPU acceleration with `n_gpu_layers`
- Process text in chunks to avoid memory bottlenecks

## Example Workflow

```bash
# Step 1: Create reference logits with high-precision model
python kl.py \
  -m models/llama-7b-fp16.gguf \
  -t datasets/wiki_test.txt \
  -w reference_logits.gz \
  -c 1024 \
  -ngl 0

# Step 2: Evaluate quantized model against reference
python kl.py \
  -m models/llama-7b-q4_0.gguf \
  -r reference_logits.gz \
  -c 1024 \
  -ngl 0
```

## Interpreting Results

### KL Divergence Values
- **Lower values** (< 0.1): Models behave very similarly
- **Medium values** (0.1-1.0): Noticeable but acceptable differences
- **Higher values** (> 1.0): Significant behavioral divergence

### Accuracy Metrics
- **High top-1 agreement** (> 90%): Very similar prediction patterns
- **Good top-5 overlap** (> 95%): Similar ranking behavior
- **Divergent metrics**: May indicate quantization artifacts or model quality differences

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce context size or batch size
2. **CUDA out of memory**: Decrease `n_gpu_layers`
3. **File format errors**: Ensure logits file was created with compatible settings
4. **Nan values**: May indicate numerical instability or model incompatibility

### Error Messages
- "Invalid header in input logit file": Corrupted or incompatible logits file
- "write_path already exists": File exists, choose different name or delete existing
- "Either text dataset or input logit file should be specified": Missing required input

## Advanced Usage

### Custom Analysis
The script saves per-token KL values to a `.kls.p` file, enabling custom statistical analysis:

```python
import pickle
import matplotlib.pyplot as plt

# Load KL values
with open('model_name.kls.p', 'rb') as f:
    kls = pickle.load(f)

# Custom analysis
plt.hist(kls, bins=50)
plt.xlabel('KL Divergence (nats)')
plt.ylabel('Frequency')
plt.title('Distribution of Token-level KL Divergence')
plt.show()
```

### Batch Processing
For large datasets, consider processing in batches and aggregating results:

```bash
# Process multiple text files
for file in datasets/*.txt; do
    python kl.py -m model.gguf -t "$file" -w "logits_${file##*/}.gz"
done
```

## References

- Original source: https://gist.github.com/Ttl/0d51f739dc59254b4b2183e259c97d82
- KL Divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- llama.cpp: https://github.com/ggerganov/llama.cpp