# Model Drift Report

**Generated:** 2026-01-12T23:14:06.091282+00:00

## Windows

- **Baseline:** 2026-01-05 to 2026-01-11
- **Candidate:** 2026-01-11T23:14:06.088754+00:00 to 2026-01-12T23:14:06.088754+00:00

## Summary

- Baseline requests: 119
- Candidate requests: 120

## Alerts

No alerts triggered.

## Metrics Comparison

### Input Drift

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| Question length (mean) | 32.6 | 32.0 | -0.5 |

### Guardrail Drift

| Metric | Baseline | Candidate | Delta (pp) |
|--------|----------|-----------|------------|
| Safety deny rate | 14.3% | 15.8% | 1.5 |
| Relevance deny rate | 25.5% | 23.8% | -1.7 |

### Agent Drift

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Agent call rate | 63.9% | 64.2% |
| Tool calls (mean) | 1.76 | 1.86 |
| Latency p95 (ms) | 1445 | 1370 |

### Error Drift

- Baseline error rate: 5.0%
- Candidate error rate: 4.2%
