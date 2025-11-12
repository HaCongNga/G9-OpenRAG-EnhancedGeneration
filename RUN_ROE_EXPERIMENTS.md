# Running RoE Experiments with Open-RAG

This guide shows you how to test RoE (Repetition of Experts) with your Open-RAG downstream tasks.

## Quick Start

### Option 1: Using the Batch Script (Windows - Easiest!)

Simply double-click `run_roe_experiments.bat` or run:

```cmd
run_roe_experiments.bat
```

You'll see an interactive menu:
```
============================================================================
RoE Experiment Runner for Open-RAG
============================================================================

Current Configuration:
  Model: shayekh/openrag_llama2_7b_8x135m
  Task: hotpotqa
  Max tokens: 100
  N docs: 3

Options:
  1) Run baseline only
  2) Run RoE only (K=8, tau=1.0)
  3) Run RoE only (K=4, tau=1.0) - Quick test
  4) Quick experiment (Baseline + RoE K=4)
  5) Standard experiment (Baseline + RoE K=8)
  6) Compare existing results
  7) Exit

Select option (1-7):
```

**Recommended for first test:** Choose option 3 (Quick test with K=4)

### Option 2: Command Line

Run baseline:
```cmd
python run_short_form_multihop_roe.py ^
    --model_name shayekh/openrag_llama2_7b_8x135m ^
    --dataset shayekh/openrag_bench ^
    --task hotpotqa ^
    --mode adaptive_retrieval ^
    --max_new_tokens 100 ^
    --threshold 0.0 ^
    --metric hotpotem ^
    --ndocs 3 ^
    --use_groundness ^
    --use_utility ^
    --use_seqscore ^
    --output_file ./eval_baseline/hotpotqa.jsonl
```

Run RoE:
```cmd
python run_short_form_multihop_roe.py ^
    --model_name shayekh/openrag_llama2_7b_8x135m ^
    --dataset shayekh/openrag_bench ^
    --task hotpotqa ^
    --mode adaptive_retrieval ^
    --max_new_tokens 100 ^
    --threshold 0.0 ^
    --metric hotpotem ^
    --ndocs 3 ^
    --use_groundness ^
    --use_utility ^
    --use_seqscore ^
    --use_roe ^
    --roe_k 8 ^
    --roe_tau 1.0 ^
    --output_file ./eval_roe/hotpotqa.jsonl
```

Compare results:
```cmd
python roe_implementation\compare_results.py ^
    eval_baseline\hotpotqa.jsonl ^
    eval_roe\hotpotqa.jsonl ^
    --num_examples 10
```

## Understanding the Parameters

### RoE-Specific Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--use_roe` | Enable RoE | Flag (add to enable) |
| `--roe_k` | Number of parallel samples | 4 (quick), 8 (standard), 16 (best) |
| `--roe_tau` | Temperature for middle layers | 0.01 (very conservative), 0.05 (recommended), 0.1 (aggressive) |

**Important:** For Open-RAG, use small tau values (0.01-0.1). Large values (>0.5) add too much noise and degrade quality!

### Expected Behavior

**Without RoE (Baseline):**
- Generation: ~100ms per token
- Deterministic expert selection
- Single forward pass per token

**With RoE (K=8):**
- Generation: ~800ms per token (8x slower)
- Stochastic expert selection (K different combinations)
- Probability averaging across K samples
- More robust predictions

## Files Created

```
openrag/
├── run_short_form_multihop_roe.py    # RoE-enhanced evaluation script
├── run_roe_experiments.bat           # Windows batch script (interactive)
├── run_roe_experiments.sh            # Linux/Mac bash script
├── eval_baseline/                    # Baseline results
│   └── hotpotqa.jsonl
├── eval_roe/                         # RoE results
│   ├── k4_tau1.0/
│   │   └── hotpotqa.jsonl
│   └── k8_tau1.0/
│       └── hotpotqa.jsonl
└── roe_implementation/               # RoE implementation
    ├── roe_openrag.py
    ├── compare_results.py
    └── ...
```

## Experiment Workflows

### Workflow 1: Quick Test (Recommended First!)

**Purpose:** Verify RoE works and see initial results quickly

**Steps:**
1. Run `run_roe_experiments.bat`
2. Choose option 3 (Quick test, K=4)
3. Wait ~4x longer than baseline
4. Results appear on screen

**Time:** ~2-4 hours for HotpotQA dev set

### Workflow 2: Standard Evaluation

**Purpose:** Full RoE evaluation with K=8

**Steps:**
1. Run `run_roe_experiments.bat`
2. Choose option 5 (Standard experiment)
3. Waits for both baseline and RoE
4. Automatic comparison at the end

**Time:** ~6-8 hours for HotpotQA dev set

### Workflow 3: Ablation Study

**Purpose:** Test different K values or tau values

**For K ablation:**
1. Edit `run_roe_experiments.bat`
2. Run baseline once
3. Run RoE with K=2, 4, 8, 16
4. Compare all results

**For tau ablation:**
1. Run baseline once
2. Run RoE with tau=0.5, 1.0, 2.0
3. Compare all results

## Reading the Results

### Metric Comparison

```
============================================================================
METRIC COMPARISON
============================================================================

Overall Score:
  Baseline: 0.4523
  RoE:      0.4891
  Improvement: +8.14%

Detailed Metrics:

  EM:
    Baseline: 0.4523
    RoE:      0.4891
    Improvement: +8.14%

  F1:
    Baseline: 0.5234
    RoE:      0.5678
    Improvement: +8.48%
```

### What to Expect

**Good signs (RoE is working):**
- ✅ EM/F1 improvement of 3-10%
- ✅ More stable predictions
- ✅ Better handling of ambiguous questions

**Red flags (something wrong):**
- ❌ No improvement or worse performance
  - Check: Is tau > 0 for middle layers?
  - Check: Is K large enough (try K=8)?
- ❌ Timeout or OOM errors
  - Reduce K (try K=4)
  - Use smaller max_new_tokens

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'roe_openrag'"

**Solution:**
```cmd
cd roe_implementation
python test_roe.py
```

If tests pass, the issue is the import path. Make sure:
1. You're running from the `openrag` root directory (where `run_short_form_multihop_roe.py` is)
2. The `roe_implementation` folder exists with `roe_openrag.py` inside
3. Try running: `python -c "import sys; sys.path.insert(0, 'roe_implementation'); import roe_openrag; print('Import successful!')"`

### Issue: Model loading fails with safetensors error

**Solution:** 
The model file is corrupted. See the fix in `run_short_form_multihop_roe.py` - it includes `resume_download=True`.

### Issue: Out of memory

**Solution:**
- Reduce K: Use `--roe_k 4` instead of 8
- Reduce ndocs: Use `--ndocs 2` instead of 3
- Reduce max_new_tokens: Use `--max_new_tokens 50` instead of 100

### Issue: Too slow

**Expected:** RoE is K times slower than baseline.

**Solutions:**
- Use K=4 for faster experiments
- Test on a subset first
- Use fewer documents (--ndocs 2)

### Issue: No improvement over baseline

**Check:**
1. Is `--use_roe` flag present?
2. Is tau > 0 for middle layers?
3. Is K >= 4?

**Try:**
- Increase K to 8 or 16
- Adjust tau (try 0.5 or 2.0)
- Check task difficulty (RoE helps more on harder tasks)

## Tips for Best Results

1. **Start Small:** Test with K=4 first to verify everything works

2. **Choose Right K:**
   - K=4: Quick experiments, ~4x slower
   - K=8: Standard evaluation, ~8x slower
   - K=16: Best quality, ~16x slower

3. **Tune tau (for Open-RAG, use small values!):**
   - tau=0.01: Very conservative, minimal noise
   - tau=0.05: Recommended default (good balance)
   - tau=0.1: More exploration (careful, may degrade quality)

4. **Monitor Progress:**
   - Results saved every 10 examples to `*_tmp` file
   - Can stop early and still get partial results

5. **Compare Fairly:**
   - Always use same seed
   - Same hyperparameters except RoE
   - Same evaluation set

## Example Output

```cmd
C:\Users\Hong\Documents\learn\openrag>python run_short_form_multihop_roe.py --use_roe --roe_k 8 --roe_tau 1.0 ...

================================================================================
RoE (Repetition of Experts) ENABLED
================================================================================
RoE K (parallel samples): 8
RoE tau (temperature): 1.0
Expected slowdown: ~8x
================================================================================

Loading model...
Model loaded successfully!

Created tau map for 32 layers
  First layer: tau=0.0
  Middle layers: tau=1.0
  Last layer: tau=0.0

Starting evaluation on 100 examples...
Output file: ./eval_roe/hotpotqa.jsonl

[RoE] Generating with K=8, max_new_tokens=100
[RoE] Step 1: Encoding prompt with clean path (tau=0)...
[RoE] Step 0: Generating with K=8 parallel paths...
[RoE]   Selected token: The
...

Progress: 10/100
Current average: 0.45
Average em: 0.45, f1: 0.52, precision: 0.58, recall: 0.48

...

================================================================================
EVALUATION COMPLETE
================================================================================
Final result: 0.4891
Retrieval Frequencies: 0.65
Average em: 0.4891, f1: 0.5678, precision: 0.6123, recall: 0.5234

RoE Configuration:
  K (samples): 8
  tau (temperature): 1.0

Results saved to: ./eval_roe/hotpotqa.jsonl
================================================================================
```

## Next Steps

After getting results:

1. **Compare results:**
   ```cmd
   python roe_implementation\compare_results.py eval_baseline\hotpotqa.jsonl eval_roe\hotpotqa.jsonl
   ```

2. **Analyze improvements:**
   - Check which examples RoE fixed
   - Look at disagreement patterns
   - Analyze retrieval decisions

3. **Try different settings:**
   - Different K values
   - Different tau values
   - Different tasks (fever, arc_c, etc.)

4. **Share findings:**
   - Document improvements
   - Note any failure cases
   - Compare to paper results

Good luck with your experiments! 🚀
