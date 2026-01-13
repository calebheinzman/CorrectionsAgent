# Evaluation Report: 20260113T014455Z

**Generated:** 2026-01-13T01:44:55.119994+00:00

## Overview

- **Datasets:** relevance, safety, tool_pairs, datapoint_pairs, question_list
- **Models:** gemini-2.0-flash-lite, gemini-2.0-flash
- **Prompt Versions:** v1.0.0, v1.1.0, v1.2.0
- **Total Samples:** 8100
- **Total Configurations:** 45

## Dataset: relevance

### Classification Metrics

| Model | Prompt | Accuracy | Precision | Recall | F1-Macro | F1-Micro |
|-------|--------|----------|-----------|--------|----------|----------|
| gemini-2.0-flash-lite | v1.0.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash-lite | v1.1.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash-lite | v1.2.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.0.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.1.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.2.0 | 0.945 | 0.000 | 0.000 | 0.500 | 1.000 |

## Dataset: safety

### Classification Metrics

| Model | Prompt | Accuracy | Precision | Recall | F1-Macro | F1-Micro |
|-------|--------|----------|-----------|--------|----------|----------|
| gemini-2.0-flash-lite | v1.0.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash-lite | v1.1.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash-lite | v1.2.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.0.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.1.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |
| gemini-2.0-flash | v1.2.0 | 0.880 | 0.000 | 0.000 | 0.500 | 1.000 |


## Dataset: tool_pairs

### Tool Selection Metrics

| Model | Prompt | Accuracy | Precision | Recall | F1-Macro | F1-Micro |
|-------|--------|----------|-----------|--------|----------|----------|
| gemini-2.0-flash-lite | v1.0.0 | 0.315 | 0.968 | 0.416 | 0.518 | 0.582 |
| gemini-2.0-flash-lite | v1.1.0 | 0.335 | 0.970 | 0.447 | 0.582 | 0.612 |
| gemini-2.0-flash-lite | v1.2.0 | 0.430 | 0.952 | 0.543 | 0.675 | 0.691 |
| gemini-2.0-flash | v1.0.0 | 0.325 | 0.944 | 0.406 | 0.531 | 0.568 |
| gemini-2.0-flash | v1.1.0 | 0.295 | 0.845 | 0.410 | 0.527 | 0.552 |
| gemini-2.0-flash | v1.2.0 | 0.370 | 0.890 | 0.522 | 0.639 | 0.658 |

## Dataset: datapoint_pairs

### Data Retrieval Metrics

| Model | Prompt | Accuracy | Success Rate |
|-------|--------|----------|--------------|
| gemini-2.0-flash-lite | v1.0.0 | 0.295 | 1.000 |
| gemini-2.0-flash-lite | v1.1.0 | 0.300 | 1.000 |
| gemini-2.0-flash-lite | v1.2.0 | 0.300 | 1.000 |
| gemini-2.0-flash | v1.0.0 | 0.525 | 1.000 |
| gemini-2.0-flash | v1.1.0 | 0.665 | 1.000 |
| gemini-2.0-flash | v1.2.0 | 0.530 | 1.000 |

## Dataset: question_list

### System-Level Metrics

| Model | Prompt | Samples | Success Rate | Avg Latency (ms) | P50 | P95 |
|-------|--------|---------|--------------|------------------|-----|-----|
| gemini-2.0-flash-lite | v1.0.0 | 100 | 1.000 | 460.9 | 581.8 | 1077.1 |
| gemini-2.0-flash-lite | v1.1.0 | 100 | 1.000 | 480.3 | 557.9 | 1370.4 |
| gemini-2.0-flash-lite | v1.2.0 | 100 | 1.000 | 523.3 | 613.5 | 1413.6 |
| gemini-2.0-flash | v1.0.0 | 100 | 1.000 | 589.9 | 716.5 | 1645.7 |
| gemini-2.0-flash | v1.1.0 | 100 | 1.000 | 767.4 | 718.5 | 2215.3 |
| gemini-2.0-flash | v1.2.0 | 100 | 1.000 | 871.7 | 813.8 | 2914.9 |



## LLM-as-Judge & RAGAS Metrics

These metrics are **scored** (not accuracy-based). Each metric is a 0-1 score representing quality:

- **Faithfulness**: Claims supported by retrieved contexts (RAGAS)
- **Context Precision**: Relevant chunks ranked early (RAGAS)
- **Completeness**: Coverage of question requirements (LLM-judged)
- **Clarity**: Structure, conciseness, investigator usefulness (LLM-judged)
- **Relevance**: On-topic, avoids filler (LLM-judged)
- **Answer Relevance (Emb)**: Generated questions similarity to original (embedding-based)
- **Context Utilization**: Cited chunks appear early in retrieval (deterministic)
- **Citation Correctness**: Cited IDs present in chunks (deterministic)

### RAGAS-Style Metrics (LLM-as-Judge)

| Model | Prompt | Samples | Faithfulness | Context Precision |
|-------|--------|---------|--------------|-------------------|
| gemini-2.0-flash-lite | v1.0.0 | 100 | 0.269 | 0.090 |
| gemini-2.0-flash-lite | v1.1.0 | 100 | 0.287 | 0.050 |
| gemini-2.0-flash-lite | v1.2.0 | 100 | 0.285 | 0.060 |
| gemini-2.0-flash | v1.0.0 | 100 | 0.292 | 0.086 |
| gemini-2.0-flash | v1.1.0 | 100 | 0.299 | 0.080 |
| gemini-2.0-flash | v1.2.0 | 100 | 0.351 | 0.099 |

### Answer Quality Metrics (LLM-as-Judge)

| Model | Prompt | Samples | Completeness | Clarity | Relevance |
|-------|--------|---------|--------------|---------|-----------|
| gemini-2.0-flash-lite | v1.0.0 | 100 | 0.275 | 0.733 | 0.289 |
| gemini-2.0-flash-lite | v1.1.0 | 100 | 0.219 | 0.757 | 0.332 |
| gemini-2.0-flash-lite | v1.2.0 | 100 | 0.236 | 0.756 | 0.334 |
| gemini-2.0-flash | v1.0.0 | 100 | 0.286 | 0.749 | 0.363 |
| gemini-2.0-flash | v1.1.0 | 100 | 0.305 | 0.761 | 0.400 |
| gemini-2.0-flash | v1.2.0 | 100 | 0.297 | 0.771 | 0.401 |


### Reference-Free Metrics (Non-Judge)

| Model | Prompt | Samples | Answer Relevance (Emb) | Context Utilization | Citation Correctness |
|-------|--------|---------|------------------------|---------------------|----------------------|
| gemini-2.0-flash-lite | v1.0.0 | 100 | 0.712 | 0.090 | 0.980 |
| gemini-2.0-flash-lite | v1.1.0 | 100 | 0.714 | 0.050 | 1.000 |
| gemini-2.0-flash-lite | v1.2.0 | 100 | 0.715 | 0.060 | 0.995 |
| gemini-2.0-flash | v1.0.0 | 100 | 0.711 | 0.110 | 0.965 |
| gemini-2.0-flash | v1.1.0 | 100 | 0.726 | 0.110 | 0.970 |
| gemini-2.0-flash | v1.2.0 | 100 | 0.735 | 0.156 | 0.955 |

---

*Report generated by Model Evaluation System*