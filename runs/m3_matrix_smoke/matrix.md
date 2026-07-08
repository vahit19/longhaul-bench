| run | operator | policy | budget | feedback_noise | episodes | exact_accuracy | memory_hit_rate | latency_p50_s | peak_system_rss_mb |
|---|---|---|---|---|---|---|---|---|---|
| frozen | frozen | fifo | 100 | 0.0 | 10 | 0.6 | 0 | 4.63 | 3867 |
| append | append | fifo | 100 | 0.0 | 10 | 0.7 | 0.5 | 3.71 | 3860 |
| reflect | reflect | compress | 100 | 0.0 | 10 | 0.7 | 0.5 | 3.9399999999999995 | 3864 |
| gated | gated | importance | 100 | 0.0 | 10 | 0.7 | 0.5 | 3.62 | 3862 |
| oracle | frozen | fifo | 100 | 0.0 | 10 | 0.7 | 1 | 3.705 | 3951 |
| noise0.4 | reflect | compress | 100 | 0.4 | 10 | 0.7 | 0.5 | 4.11 | 3998 |
