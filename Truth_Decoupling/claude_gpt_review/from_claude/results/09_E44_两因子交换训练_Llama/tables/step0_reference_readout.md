## reference ranking (max-true margin / any-true top1 are not standard MC1) and log p("I don't know.")
| family | layer | arm | split | scored | any-true top1 | MC2 | max-true margin change | to correct / to wrong | log p(abstain) change |
|---|---|---|---|---|---|---|---|---|---|
| — | — | no write, caut | train | 93 | 0.6452 | 0.6266 | — | — | (level -7.0403) |
| — | — | no write, caut | tune | 192 | 0.5260 | 0.5153 | — | — | (level -8.1083) |
| — | — | no write, caut_E1 | train | 93 | 0.6559 | 0.6622 | — | — | (level -5.8961) |
| — | — | no write, caut_E1 | tune | 192 | 0.5729 | 0.5646 | — | — | (level -6.6498) |
| — | — | no write, caut_E2 | train | 93 | 0.7312 | 0.6994 | — | — | (level -5.3239) |
| — | — | no write, caut_E2 | tune | 192 | 0.5990 | 0.6004 | — | — | (level -5.9689) |
| — | — | no write, caut_M | train | 93 | 0.4946 | 0.4839 | — | — | (level -6.5324) |
| — | — | no write, caut_M | tune | 192 | 0.4062 | 0.4036 | — | — | (level -7.2635) |
| — | — | no write, conf | train | 93 | 0.5699 | 0.5765 | — | — | (level -24.4147) |
| — | — | no write, conf | tune | 192 | 0.5104 | 0.4978 | — | — | (level -24.8193) |
| — | — | no write, conf_E1 | train | 93 | 0.6237 | 0.6375 | — | — | (level -23.2677) |
| — | — | no write, conf_E1 | tune | 192 | 0.5521 | 0.5484 | — | — | (level -23.2481) |
| — | — | no write, conf_E2 | train | 93 | 0.6344 | 0.6461 | — | — | (level -21.7261) |
| — | — | no write, conf_E2 | tune | 192 | 0.5833 | 0.5827 | — | — | (level -21.7019) |
| — | — | no write, conf_M | train | 93 | 0.5054 | 0.5036 | — | — | (level -20.5526) |
| — | — | no write, conf_M | tune | 192 | 0.4323 | 0.4266 | — | — | (level -20.7367) |
| — | — | no write, x | train | 93 | 0.6452 | 0.6268 | — | — | (level -6.9776) |
| — | — | no write, x | tune | 192 | 0.5312 | 0.5275 | — | — | (level -8.1059) |
| — | — | no write, x_E1 | train | 93 | 0.6667 | 0.6620 | — | — | (level -5.9487) |
| — | — | no write, x_E1 | tune | 192 | 0.5729 | 0.5717 | — | — | (level -6.6650) |
| — | — | no write, x_F | train | 93 | 0.6237 | 0.6042 | — | — | (level -6.8346) |
| — | — | no write, x_F | tune | 192 | 0.5104 | 0.5056 | — | — | (level -8.0803) |
