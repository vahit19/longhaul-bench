| run | operator | policy | budget | feedback_noise | episodes | exact_accuracy | memory_hit_rate | latency_p50_s | peak_system_rss_mb |
|---|---|---|---|---|---|---|---|---|---|
| frozen | frozen | fifo | 100 | 0.0 | 945 | 0.17142857142857143 | 0 | 5.5 | 11516 |
| reflect | reflect | compress | 100 | 0.0 | 945 | 0.491005291005291 | 0.9883597883597883 | 7.0 | 11519 |
| noise0.4 | reflect | compress | 100 | 0.4 | 945 | 0.4105820105820106 | 0.9873015873015873 | 6.63 | 11491 |
| defended0.4 | reflect | compress | 100 | 0.4 | 945 | 0.4222222222222222 | 0.9386243386243386 | 6.49 | 11499 |
