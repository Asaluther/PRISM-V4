# 夜间 P 队列执行报告

- 完成时间: 2026-09-09 23:11:03

## P1b 稳定矩阵（K x clip x lr_loc）

| 配置 | NaN 数 | best PPL | 稳定 |
|---|---|---|---|
| p1b_K2_c0.25_l1.5e-4 | 0 | 57.50 | no |
| p1b_K2_c0.25_l5e-4 | 0 | inf | no |
| p1b_K2_c0.5_l1.5e-4 | 0 | 51.25 | no |
| p1b_K2_c0.5_l5e-4 | 0 | 86.47 | no |
| p1b_K3_c0.25_l1.5e-4 | 0 | 55.05 | no |
| p1b_K3_c0.25_l5e-4 | 0 | 81.87 | no |
| p1b_K3_c0.5_l1.5e-4 | 0 | 49.63 | no |
| p1b_K3_c0.5_l5e-4 | 0 | 69.64 | no |

## P1b 终判（若稳定配置存在）

- m1-inf: 未产出（无稳定配置）
- m2-inf: 未产出（无稳定配置）

## P3 门来源 2x2（PCN 内部开关）

- 校准格 feat/wlat_e: [0.309, 0.167, 0.165]  valid=True
- feat/wlat_e(control): mean=0.214 copies=[0.309, 0.167, 0.165]
- qk/wlat_e: mean=0.092 copies=[0.114, 0.072, 0.091]
- feat/wlat_h: mean=0.022 copies=[0.026, 0.02, 0.02]
- qk/wlat_h: mean=0.047 copies=[0.047, 0.049, 0.046]

---
完整日志: results/overnight_p.log | 系统即将关机（用户已授权）
