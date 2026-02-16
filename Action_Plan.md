# LLM Fine-Tuning Debugging & Optimization Action Plan

## Project Context

This project focuses on fine-tuning a GPT-2 style causal language model using raw stories and poems sourced from Hugging Face datasets. The primary objective is to enhance training stability, reduce perplexity, and ultimately improve the quality of generated stories and poems.

| Aspect | Detail |
| :----- | :----- |
| **Dataset** | Raw stories & poems (unlabeled) |
| **Source** | Hugging Face datasets |
| **Model** | Various LLMs |
| **Objective** | Improve training stability, reduce perplexity, and enhance story/poem generation quality |

## Overall Timeline Strategy

The project is structured into several stages, with an estimated total duration of 3–4 weeks, emphasizing an iterative approach.

| Stage | Type | Estimated Duration |
| :-------------------------------- | :------------------ | :----------------- |
| Data Audit & Cleaning | Diagnostic | 2–3 days |
| Pipeline Verification | Critical Validation | 1–2 days |
| Hyperparameter Stabilization | Experimental | 3–5 days |
| Structured Formatting Experiments | Improvement | 3–4 days |
| Full Training & Evaluation | Controlled Runs | 5–10 days |
| Final Evaluation & Reporting | Analysis | 3–5 days |

**Total Expected Duration:** 3–4 weeks (iterative)

## Phase-Based Action Plan

### Phase 1 — Dataset Size & Distribution Audit

*   **Duration:** 1–2 days
*   **Objective:** Ensure sufficient and balanced data.
*   **Actions:**
    *   Count total documents.
    *   Count total tokens.
    *   Compute average document length.
    *   Plot length distribution.
    *   Remove very short (<30 tokens) samples.
*   **Evaluation Method:**
    *   Token statistics analysis.
    *   Histogram of sequence lengths.
*   **Gate to Move Forward:**
    *   Dataset ≥ several million tokens (or justified small experiment).
    *   No extreme imbalance in document lengths.

### Phase 2 — Data Quality Inspection

*   **Duration:** 1–2 days
*   **Objective:** Validate text cleanliness and formatting.
*   **Actions:**
    *   Print 50 random samples.
    *   Check punctuation preservation.
    *   Check casing consistency.
    *   Remove HTML/noise.
    *   Remove duplicates.
*   **Evaluation Method:**
    *   Manual qualitative review.
    *   Random sampling verification.
*   **Gate to Move Forward:**
    *   Text reads naturally.
    *   No formatting corruption.

### Phase 3 — Tokenization Verification

*   **Duration:** 0.5–1 day
*   **Objective:** Ensure tokenizer-model compatibility.
*   **Actions:**
    *   Encode sample text.
    *   Decode it back.
    *   Compare reconstructed text.
*   **Evaluation Method:**
    *   Reconstruction similarity check.
    *   Inspect token counts.
*   **Gate to Move Forward:**
    *   Decoded ≈ original.
    *   No abnormal token expansion.

### Phase 4 — Micro-Overfit Test (Pipeline Validation)

*   **Duration:** 1 day
*   **Objective:** Confirm training pipeline correctness.
*   **Actions:**
    *   Train on 100 samples.
    *   Run 5–10 epochs.
*   **Evaluation Method:**
    *   Monitor training loss curve.
*   **Gate to Move Forward:**
    *   Loss < 1.5.
    *   Clear overfitting observed.
    *   *If this fails → STOP and debug pipeline.* 

### Phase 5 — Hyperparameter Stabilization

*   **Duration:** 3–5 days
*   **Objective:** Achieve stable optimization.
*   **Actions:**
    *   Test learning rates: 5e-5, 2e-5, 1e-5.
    *   Adjust batch size.
    *   Use gradient accumulation.
*   **Evaluation Method:**
    *   Smooth loss curve.
    *   No sudden spikes.
    *   Stable gradient norms.
*   **Gate to Move Forward:**
    *   Loss decreases consistently in first 1000 steps.

### Phase 6 — Padding & Label Masking Verification

*   **Duration:** 1 day
*   **Objective:** Ensure correct loss calculation.
*   **Actions:**
    *   Confirm padding tokens ignored in loss.
    *   Confirm correct label shifting.
*   **Evaluation Method:**
    *   Inspect training loop.
    *   Verify percentage of padding tokens.
*   **Gate to Move Forward:**
    *   Padding excluded from loss.
    *   No abnormal perplexity inflation.

### Phase 7 — Sequence Length Optimization

*   **Duration:** 2–3 days
*   **Objective:** Improve narrative learning.
*   **Actions:**
    *   Compare `max_length`: 128, 256, 512.
*   **Evaluation Method:**
    *   Validation perplexity comparison.
    *   Coherence evaluation.
*   **Gate to Move Forward:**
    *   Lower validation perplexity.
    *   Better long-range coherence.

### Phase 8 — Structured Formatting Injection

*   **Duration:** 3–4 days
*   **Objective:** Improve stylistic consistency.
*   **Actions:**
    *   Add tags: `<STORY>`, `<POEM>`.
    *   Standardize formatting.
*   **Evaluation Method:**
    *   Human qualitative review.
    *   Prompt-based generation comparison.
*   **Gate to Move Forward:**
    *   Output style becomes consistent.
    *   Reduced genre mixing.

### Phase 9 — Full Training with Validation Split

*   **Duration:** 5–10 days
*   **Objective:** Controlled training experiment.
*   **Actions:**
    *   90/10 train-validation split.
    *   Monitor both losses.
    *   Early stopping.
*   **Evaluation Method:**
    *   Compare train vs validation curves.
    *   Track perplexity.
*   **Success Criteria:**
    *   Validation PPL < 80 (small dataset acceptable).
    *   Stable convergence.

### Phase 10 — Final Evaluation

#### Quantitative Metrics

*   Validation Perplexity
*   Repetition Rate
*   Self-BLEU (diversity)

#### Qualitative Metrics

*   Coherence (1–5)
*   Fluency (1–5)
*   Originality (1–5)

#### Acceptance Criteria

*   Significant improvement over baseline GPT-2.
*   Reduced repetition loops.
*   Human score ≥ 3.5 average.

## Iteration Logic

If any phase fails:

*   Return to previous diagnostic phase.
*   Document findings.
*   Run controlled re-experiment.

All experiments must be logged with:

*   Dataset version
*   Hyperparameters
*   Model size
*   Validation metrics
*   Generation samples

## Important Note on Labels

Unlabeled raw text is correct for causal language modeling, as labels are automatically generated through next-token prediction. However, inconsistent formatting, mixed genres without structure, or a small dataset size can significantly degrade performance.

## Final Timeline Recommendation

This project should be executed in 2–3 experimental cycles, rather than one long training attempt. Each cycle should involve:

*   1 controlled variable change.
*   1 measurable evaluation.
*   1 documented conclusion.

**Estimated Total Project Duration:** 3–4 weeks
