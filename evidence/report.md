# Evidence

## 1. The bot is not the truth
- Bot queue == team that finally closed the request: **77.2%** (2,471 of 10,822 requests went to the wrong first team).
- Requests that needed >=1 transfer: 24.9%; average transfers per request 0.36.
- Billing queue: 26% of it is requests that merely say 'paid'; only 0.2% of those really ended in Billing (policy s3: Billing = the payment itself).
- Water purifier requests in Filters & Consumables: 53% of that queue; only 28% really belong there (other products: 96%).
- A model trained to copy the bot labels reproduces them ~96% of the time but is right about the real destination only ~77% of the time. Matching labels at 90% is achievable and is the wrong goal.

## 2. Validation (always train on the past, test on the future)
| train_to   | validate         |    n |   acc_vs_final |   acc_vs_bot_label |   bot_vs_final |   auto_coverage |   auto_acc |
|:-----------|:-----------------|-----:|---------------:|-------------------:|---------------:|----------------:|-----------:|
| 2025-10    | 2025-10..2026-01 | 2199 |          0.837 |              0.779 |          0.775 |           0.842 |      0.958 |
| 2026-01    | 2026-01..2026-04 | 2168 |          0.847 |              0.776 |          0.775 |           0.854 |      0.959 |
| 2026-04    | 2026-04..2026-07 | 2135 |          0.838 |              0.786 |          0.767 |           0.842 |      0.957 |

Mean accuracy vs real destination: **84.1%** (range 83.7%-84.7%). The bot scores 77.2% on the same rows.

## 3. Held-out Apr-Jun 2026, per team
|                       |   precision |   recall |   f1-score |   support |
|:----------------------|------------:|---------:|-----------:|----------:|
| Billing               |       0.852 |    0.859 |      0.855 |   255     |
| Filters & Consumables |       0.842 |    0.826 |      0.834 |   207     |
| Installs & Demo       |       0.837 |    0.832 |      0.834 |   321     |
| Product Advice        |       0.842 |    0.792 |      0.816 |   269     |
| Repairs               |       0.81  |    0.888 |      0.847 |   498     |
| Returns & Replacement |       0.873 |    0.819 |      0.845 |   326     |
| Warranty Claims       |       0.837 |    0.811 |      0.824 |   259     |
| accuracy              |       0.838 |    0.838 |      0.838 |     0.838 |
| macro avg             |       0.842 |    0.832 |      0.836 |  2135     |
| weighted avg          |       0.839 |    0.838 |      0.838 |  2135     |

## 4. Auto-route vs. send to a human (confidence threshold)
|   threshold |   auto_routed |   accuracy_of_auto_routed |   to_human |
|------------:|--------------:|--------------------------:|-----------:|
|         0   |         1     |                     0.838 |      0     |
|         0.4 |         0.883 |                     0.927 |      0.117 |
|         0.5 |         0.862 |                     0.944 |      0.138 |
|         0.6 |         0.842 |                     0.957 |      0.158 |
|         0.7 |         0.83  |                     0.963 |      0.17  |
|         0.8 |         0.815 |                     0.965 |      0.185 |

## 5. Where it fails
- 16% of requests are low-confidence; the model is right on only 21% of them. Typical text: 'not happy with the purifier', 'please call back regarding my fan', 'need help with my mixer'. No team can be inferred; a person has to ask.
- Confident predictions are wrong 4.3% of the time (78 of 1797).
- Exact duplicates in training (same text/product/warranty) end in the same team only ~92% of the time, so ~8% of outcomes look like noise nobody can predict from the text.
- Metadata (channel, warranty status, product) does not change the destination mix, so the model uses text only.
- Biggest confident mix-ups (real -> predicted): Warranty Claims->Returns & Replacement (5); Installs & Demo->Returns & Replacement (4); Filters & Consumables->Repairs (4); Repairs->Returns & Replacement (4); Repairs->Installs & Demo (4)

## 6. Busiest teams (for headcount)
|                       |   bot_queue |   closed_by |   transfers_out |   transfers_in |   share_of_closed |
|:----------------------|------------:|------------:|----------------:|---------------:|------------------:|
| Repairs               |        3129 |        2534 |            1136 |            541 |             0.234 |
| Billing               |        1656 |        1318 |             585 |            247 |             0.122 |
| Filters & Consumables |        1458 |        1066 |             586 |            194 |             0.099 |
| Installs & Demo       |        1185 |        1596 |              39 |            450 |             0.147 |
| Product Advice        |        1180 |        1390 |              45 |            255 |             0.128 |
| Returns & Replacement |        1174 |        1539 |              42 |            407 |             0.142 |
| Warranty Claims       |        1040 |        1379 |              38 |            377 |             0.127 |

## 7. Rupees
- Volume: 725/month (train averages 721/month; test 726/month).
- Correctly routed request costs Rs 13 on average (0.042 transfers x Rs 305); a misrouted one costs Rs 698 (1.44 transfers x Rs 305 + Rs 260 extra contact).
- Bot today: 22.8% misrouted -> Rs 169/request -> Rs 122,770/month, plus licence Rs 26,667/month. **Total Rs 149,436/month.**
- Model, route everything: 15.9% misrouted -> Rs 122/request -> Rs 88,495/month + hosting Rs 1,000. **Total Rs 89,495/month.**
- Model + triage: auto-route 85% at 4.2% error, send 15% to a human who asks one question (assumed Rs 260 each, no transfer) -> Rs 75/request -> Rs 54,562/month + hosting. **Total Rs 55,562/month.**
- Saving vs bot, route everything: Rs 59,941/month = Rs 7.2 lakh/year.
- Saving vs bot, model + triage: Rs 93,875/month = Rs 11.3 lakh/year.
- Of which licence alone: Rs 3.2 lakh/year. Prediction cost: Rs 0 (local CPU, no API); month at 725 requests = Rs 0 in API fees.