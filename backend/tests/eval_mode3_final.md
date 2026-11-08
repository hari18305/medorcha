# MedOrchestrate Evaluation Report

## Summary

| Metric | Value |
|---|---|
| Cases tested | 17 |
| Error rate (%) | 0.0 |
| RAG retrieval success (%, keyword-overlap heuristic) | 82.4 |
| Average confidence | 0.7364 |
| Optimizer activation rate (%) | 52.9 |
| HITL rate -- review_required (%) | 23.5 |
| Average latency (s) | 25.6485 |
| Average optimizer iterations | 1.0 |
| Max optimizer iterations | 3 |
| Evidence coverage (%, >=1 grounded citation) | 100.0 |
| Average evidence items | 11.4706 |
| Live evidence available (%) | 64.7 |
| Conflict detected (%) | 17.6 |

## Per-case results

| Case | Success | Confidence | Review required | Iterations | Latency (s) | RAG relevant | Evidence coverage | Live evidence | Error |
|---|---|---|---|---|---|---|---|---|---|
| CASE001 | True | 0.8159510535591685 | False | 0 | 41.99 | True | True | True |  |
| CASE001 | True | 0.8263325540458426 | False | 1 | 61.83 | True | True | False |  |
| CASE002_text_only | True | 0.8159510535591685 | False | 0 | 0.28 | True | True | True |  |
| CASE003_image_only | True | 0.47586484590414835 | True | 3 | 125.21 | False | True | True |  |
| CASE004_text_and_image | True | 0.8159510535591685 | False | 0 | 0.95 | True | True | True |  |
| CASE005_text_and_medical_report | True | 0.8159510535591685 | False | 0 | 0.28 | True | True | True |  |
| CASE006_medical_report_only | True | 0.6258648459041483 | True | 3 | 0.81 | False | True | False |  |
| CASE007_high_confidence_candidate | True | 0.7561038526554544 | False | 0 | 25.13 | True | True | True |  |
| CASE008_low_confidence_candidate | True | 0.5728438153631193 | True | 3 | 21.11 | True | True | True |  |
| CASE009_elderly_patient | True | 0.8479373411502766 | False | 1 | 5.49 | True | True | False |  |
| CASE010_pediatric_patient | True | 0.7843348600477384 | False | 0 | 24.01 | True | True | True |  |
| CASE011_with_labs | True | 0.9043325540458427 | False | 1 | 29.88 | True | True | True |  |
| CASE012_extensive_history | True | 0.6763325540458426 | False | 1 | 3.1 | True | True | False |  |
| CASE013_minimal_no_content | True | 0.6258648459041483 | True | 3 | 0.8 | False | True | False |  |
| CASE014_headache_focused | True | 0.6699198279299394 | False | 0 | 30.64 | True | True | True |  |
| CASE015_dizziness_focused | True | 0.7830465848750736 | False | 0 | 18.85 | True | True | True |  |
| CASE016_multiple_symptoms_with_image | True | 0.7063325540458426 | False | 1 | 45.66 | True | True | False |  |
