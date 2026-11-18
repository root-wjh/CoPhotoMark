# Held-out validation calibration

Config `2f9d2e8ea5fa9016`; tag length B = 48 bits.

| Unit | budget | N_faces | tau | gamma | bound | measured FAR | measured FRR | samples (pos/neg) |
|---|---|---|---|---|---|---|---|---|
| auto | 1e-14 | 1 | 0 | 1.0000 | 3.55e-15 | 0.0000 (0/450) | 0.0500 | 180/450 |
| blind | 1e-05 | 1 | 9 | 0.8125 | 7.61e-06 | 0.0000 (0/450) | 0.0944 | 180/450 |
| set | 2e-04 | 20 | 9 | 0.8125 | 7.61e-06 | 0.0000 (0/160) | 0.2000 | 40/160 |

Image-level decisions at the frozen thresholds:

| Mode | gamma | clean acceptance | genuine acceptance | image FAR |
|---|---|---|---|---|
| auto | 1.0000 | 1.0000 | 0.8000 | 0.0000 |
| blind | 0.8125 | 0.6000 | 0.6000 | 0.0000 |