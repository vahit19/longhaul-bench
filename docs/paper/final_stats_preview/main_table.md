| arm | worlds | mean acc. | bootstrap 95% CI |
|---|---|---|---|
| frozen | 5 | 58.2% | [57.4%, 58.8%] |
| append | 5 | 62.0% | [61.1%, 63.5%] |
| reflect | 5 | 61.3% | [60.2%, 62.2%] |
| gated | 3 | 61.2% | [57.3%, 64.8%] |
| oracle | 3 | 77.0% | [75.0%, 80.3%] |
| noise0.4 | 5 | 56.6% | [54.9%, 58.5%] |
| defended0.4 | 4 | 61.1% | [60.5%, 61.7%] |
| noise0.2-plausible | 1 | 55.8% | [55.8%, 55.8%] |
| noise0.4-plausible | 1 | 53.6% | [53.6%, 53.6%] |
| framing-unverified | 4 | 62.7% | [61.6%, 63.9%] |

| contrast | claim | mean diff | MWU p | Holm-adj p |
|---|---|---|---|---|
| append vs frozen | learning lifts accuracy over control | +3.8% | 0.008 | 0.032 |
| noise0.4 vs reflect | plausible corruption degrades learning (rot) | -4.7% | 0.016 | 0.048 |
| defended0.4 vs noise0.4 | consistency gates recover corrupted learning | +4.4% | 0.016 | 0.048 |
| framing-unverified vs reflect | memory framing label has no effect (null expected) | +1.4% | 0.111 | 0.111 |