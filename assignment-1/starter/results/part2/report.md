# Part II — AI/ML results

## Inputs and targets

The stage reads only the two required ML Parquet tables. The syndrome
table supplies 16-value syndrome features, a logical-error label, and a
physical sample weight. The Google table supplies shot-level labels,
four supplied decoder predictions, detector-event counts, and packed
detector bits.

- Syndrome examples: 75,598
- Google examples: 250,000
- Supplied partitions: train / validation / test.

## Task A — weighted syndrome decoder

The prior is the weighted positive-label rate on the training rows. The
linear model is logistic regression on the 16 values returned by the
supplied `syndrome_model_input` helper. Physical `sample_weight` is used
for fitting and for every Task A test metric. The classification threshold
is selected from validation data only.

| Model | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s |
|---|---:|---:|---:|---:|---:|
| task_a_prior | 0.103950 | 0.500000 | 0.096910 | 0.000000 | n/a |
| task_a_logistic | 0.220526 | 0.869649 | 0.090956 | 0.169076 | 0.001047 |

## Task B — Google decoders and linear combination

Distance three and distance five are evaluated independently. Every
supplied decoder is evaluated on the same test shots within its distance.
The combined model uses exactly the five values returned by the supplied
`google_meta_model_input` helper: normalized detector-event density plus
the four decoder predictions. Its threshold is selected from validation
data only.

### Decoder-error overlap

#### Distance 3

| Model | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s |
|---|---:|---:|---:|---:|---:|
| task_b_d3_prior | 0.492950 | 0.500000 | 0.249952 | 0.000000 | 0.000135 |
| task_b_d3_decoder_1 | 0.401330 | 0.598536 | n/a | 0.000000 | 0.012344 |
| task_b_d3_decoder_2 | 0.421970 | 0.577921 | n/a | 0.000000 | 0.012135 |
| task_b_d3_decoder_3 | 0.436360 | 0.563456 | n/a | 0.000000 | 0.012485 |
| task_b_d3_decoder_4 | 0.396680 | 0.603120 | n/a | 0.000000 | 0.012383 |
| task_b_d3_combined | 0.397980 | 0.601470 | 0.235152 | 0.143906 | 0.003878 |

Pairwise test-error overlap:

| Decoder pair | Shared errors | Jaccard overlap |
|---|---:|---:|
| belief_matching_prediction__correlated_matching_prediction | 26752 | 0.4813 |
| belief_matching_prediction__pymatching_prediction | 25442 | 0.4362 |
| belief_matching_prediction__tensor_network_contraction_prediction | 27770 | 0.5337 |
| correlated_matching_prediction__pymatching_prediction | 29814 | 0.5322 |
| correlated_matching_prediction__tensor_network_contraction_prediction | 23299 | 0.3978 |
| pymatching_prediction__tensor_network_contraction_prediction | 22190 | 0.3631 |

#### Distance 5

| Model | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s |
|---|---:|---:|---:|---:|---:|
| task_b_d5_prior | 0.501520 | 0.500000 | 0.250005 | 0.000000 | 0.000050 |
| task_b_d5_decoder_1 | 0.399440 | 0.600572 | n/a | 0.000000 | 0.004248 |
| task_b_d5_decoder_2 | 0.422600 | 0.577410 | n/a | 0.000000 | 0.004307 |
| task_b_d5_decoder_3 | 0.454800 | 0.545185 | n/a | 0.000000 | 0.004112 |
| task_b_d5_decoder_4 | 0.395240 | 0.604761 | n/a | 0.000000 | 0.004853 |
| task_b_d5_combined | 0.397280 | 0.603023 | 0.234060 | 0.035839 | 0.000933 |

Pairwise test-error overlap:

| Decoder pair | Shared errors | Jaccard overlap |
|---|---:|---:|
| belief_matching_prediction__correlated_matching_prediction | 6002 | 0.4125 |
| belief_matching_prediction__pymatching_prediction | 5762 | 0.3695 |
| belief_matching_prediction__tensor_network_contraction_prediction | 6524 | 0.4889 |
| correlated_matching_prediction__pymatching_prediction | 6381 | 0.4102 |
| correlated_matching_prediction__tensor_network_contraction_prediction | 5552 | 0.3728 |
| pymatching_prediction__tensor_network_contraction_prediction | 5305 | 0.3327 |

## Task C — bounded raw-detector prototype

The MLP consumes only the fixed subset `distance = 3 AND shot_index <
12_500`. The packed detector bytes are unpacked through the supplied
little-endian helper, producing 200 binary inputs per shot. This is a
bounded consumer check rather than an architecture comparison.

| Model | Logical-error rate | Balanced accuracy | Brier score | Training s | Prediction s |
|---|---:|---:|---:|---:|---:|
| task_c_raw_mlp | 0.500600 | 0.499379 | 0.358392 | 8.877698 | 0.012143 |

A flat 200-bit representation does not explicitly encode detector
position, spatial neighborhood, or the relationship between detector
events across QEC rounds. The MLP therefore receives the right raw
bits but not an explicit geometry/time structure; any such structure
must be inferred indirectly from repeated bit positions.

## Limitations

The experiments are intentionally simple and bounded. The combined
decoder is linear, the syndrome representation is the supplied flat
16-value helper output, and Task C uses a small MLP on a fixed d=3
subset. Numerical performance is not the grading target; correctness
of prepared-data use, supplied splits, physical weighting, repeatability,
evaluation, and traceable saved outputs is the focus.
