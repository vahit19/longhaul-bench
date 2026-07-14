## Per-arm accuracy (world-level, 5 seeds unless noted)

| arm | worlds | mean | bootstrap 95% CI |
|---|---|---|---|
| frozen | 5 | 58.2% | [57.4%, 58.8%] |
| append | 5 | 62.0% | [61.1%, 63.5%] |
| reflect | 5 | 61.3% | [60.2%, 62.2%] |
| gated | 5 | 62.9% | [59.7%, 65.3%] |
| oracle | 5 | 76.8% | [74.9%, 78.9%] |
| noise0.2 | 5 | 59.0% | [57.3%, 60.2%] |
| noise0.4 | 5 | 55.3% | [54.1%, 56.5%] |
| defended0.4 | 5 | 61.3% | [60.6%, 62.0%] |
| framing-unverified | 4 | 62.7% | [61.6%, 63.9%] |
| noise0.4-crude | 1 | 60.0% | [60.0%, 60.0%] |

## Confirmatory contrasts (Mann-Whitney U, Holm-corrected family)

| contrast | claim | Δmean | Cliff's δ | MWU p | Holm-adj p | verdict |
|---|---|---|---|---|---|---|
| append vs frozen | learning lifts accuracy over frozen control | +3.8% | +1.00 | 0.008 | 0.048 | significant |
| oracle vs frozen | manual-oracle ceiling above control | +18.6% | +1.00 | 0.008 | 0.048 | significant |
| noise0.4 vs reflect | plausible corruption degrades learning (rot) | -6.0% | -1.00 | 0.008 | 0.048 | significant |
| noise0.2 vs reflect | rot at lower dose | -2.3% | -0.76 | 0.056 | 0.111 | n.s. |
| defended0.4 vs noise0.4 | consistency gates recover corrupted learning | +6.0% | +1.00 | 0.008 | 0.048 | significant |
| framing-unverified vs reflect | memory-framing label has no effect (null) | +1.4% | +0.70 | 0.111 | 0.111 | null (expected) |