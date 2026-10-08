# Tests

```bash
pytest -m "not slow"     # fast suite
pytest -m slow           # convergence and end-to-end tests
pytest --cov=adelmo
```

## Markers

| Marker | Meaning |
|---|---|
| `slow` | Computationally expensive |
| `regression` | Fast behavioural regression contracts |
| `integration` | Multi-component integration |
| `convergence` | Full optimiser convergence |
| `e2e` | Full machine-to-optimisation end to end |
| `lhc`, `sps`, `psb` | Machine-specific |
| `serial` | Must not run in parallel (do not combine with `-n`) |

## Layout

| Directory | Contents |
|---|---|
| `accelerators/`, `mad/` | Machine definitions and MAD-NG interfaces |
| `training/` | Fitters end to end; these serve as worked examples |
| `workers/` | Worker processes |
| `measurements/`, `model_creator/` | Measurement preparation and model construction |
| `optimisers/`, `analysis/` | Optimisers and degeneracy analysis |
| `data/` | Input data and model files |

Unit tests for BPM phase utilities are in the `tmom-recon` repository.
