# Error Analysis
Generated: 2025-12-22T07:18:44.278100Z
Errors dir: outputs/results/run_20251221_003516/errors

## Model: ast_transfer
- errors: 20 / 400 (acc 0.9500)
- top true labels: helicopter (5), fireworks (2), brushing teeth (2), door wood creaks (1), wind (1)
- top predicted labels: footsteps (3), airplane (3), washing machine (2), hand saw (2), snoring (2)
- top confusions: helicopter -> airplane (3), fireworks -> footsteps (2), helicopter -> washing machine (2), brushing teeth -> hand saw (2), door wood creaks -> cat (1)
- hardest labels: helicopter (5/8, 0.62), brushing teeth (2/8, 0.25), fireworks (2/8, 0.25), breathing (1/8, 0.12), can opening (1/8, 0.12)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/ast_transfer

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| helicopter | 5 | 8 | 0.62 | airplane (3) |
| brushing teeth | 2 | 8 | 0.25 | hand saw (2) |
| fireworks | 2 | 8 | 0.25 | footsteps (2) |
| breathing | 1 | 8 | 0.12 | snoring (1) |
| can opening | 1 | 8 | 0.12 | insects (1) |
| coughing | 1 | 8 | 0.12 | glass breaking (1) |
| cow | 1 | 8 | 0.12 | snoring (1) |
| crackling fire | 1 | 8 | 0.12 | crickets (1) |
| door wood creaks | 1 | 8 | 0.12 | cat (1) |
| door wood knock | 1 | 8 | 0.12 | footsteps (1) |

## Model: clap_transfer
- errors: 11 / 400 (acc 0.9725)
- top true labels: helicopter (4), crickets (2), can opening (2), pig (1), footsteps (1)
- top predicted labels: washing machine (2), frog (2), rooster (1), airplane (1), wind (1)
- top confusions: helicopter -> washing machine (2), crickets -> frog (2), pig -> rooster (1), helicopter -> airplane (1), helicopter -> wind (1)
- hardest labels: helicopter (4/8, 0.50), can opening (2/8, 0.25), crickets (2/8, 0.25), chirping birds (1/8, 0.12), footsteps (1/8, 0.12)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/clap_transfer

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| helicopter | 4 | 8 | 0.50 | washing machine (2) |
| can opening | 2 | 8 | 0.25 | glass breaking (1) |
| crickets | 2 | 8 | 0.25 | frog (2) |
| chirping birds | 1 | 8 | 0.12 | crickets (1) |
| footsteps | 1 | 8 | 0.12 | door wood knock (1) |
| pig | 1 | 8 | 0.12 | rooster (1) |

## Model: clap_zeroshot
- errors: 34 / 400 (acc 0.9150)
- top true labels: wind (5), insects (5), hen (5), washing machine (4), helicopter (4)
- top predicted labels: Sound of airplane (10), Sound of rooster (7), Sound of train (4), Sound of cow (3), Sound of frog (2)
- top confusions: hen -> Sound of rooster (5), washing machine -> Sound of airplane (3), wind -> Sound of train (3), helicopter -> Sound of airplane (3), wind -> Sound of airplane (2)
- hardest labels: hen (5/8, 0.62), insects (5/8, 0.62), wind (5/8, 0.62), helicopter (4/8, 0.50), washing machine (4/8, 0.50)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/clap_zeroshot

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| hen | 5 | 8 | 0.62 | Sound of rooster (5) |
| insects | 5 | 8 | 0.62 | Sound of cow (2) |
| wind | 5 | 8 | 0.62 | Sound of train (3) |
| helicopter | 4 | 8 | 0.50 | Sound of airplane (3) |
| washing machine | 4 | 8 | 0.50 | Sound of airplane (3) |
| crickets | 2 | 8 | 0.25 | Sound of frog (2) |
| drinking sipping | 2 | 8 | 0.25 | Sound of pig (1) |
| pig | 2 | 8 | 0.25 | Sound of rooster (1) |
| can opening | 1 | 8 | 0.12 | Sound of glass breaking (1) |
| clock tick | 1 | 8 | 0.12 | Sound of mouse click (1) |

## Model: cnn_fold5
- errors: 100 / 400 (acc 0.7500)
- top true labels: fireworks (8), pig (6), washing machine (5), door wood creaks (5), breathing (5)
- top predicted labels: keyboard typing (9), drinking sipping (5), cow (4), car horn (4), cat (4)
- top confusions: fireworks -> keyboard typing (6), pig -> cow (2), wind -> car horn (2), crackling fire -> washing machine (2), frog -> sheep (2)
- hardest labels: fireworks (8/8, 1.00), pig (6/8, 0.75), breathing (5/8, 0.62), door wood creaks (5/8, 0.62), washing machine (5/8, 0.62)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/cnn_fold5

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| fireworks | 8 | 8 | 1.00 | keyboard typing (6) |
| pig | 6 | 8 | 0.75 | cow (2) |
| breathing | 5 | 8 | 0.62 | snoring (2) |
| door wood creaks | 5 | 8 | 0.62 | sheep (1) |
| washing machine | 5 | 8 | 0.62 | engine (1) |
| frog | 4 | 8 | 0.50 | sheep (2) |
| helicopter | 4 | 8 | 0.50 | crackling fire (1) |
| laughing | 4 | 8 | 0.50 | crying baby (2) |
| water drops | 4 | 8 | 0.50 | can opening (1) |
| airplane | 3 | 8 | 0.38 | wind (2) |

## Model: llm_predictions
- errors: 88 / 400 (acc 0.7800)
- top true labels: wind (8), train (8), frog (6), water drops (6), helicopter (5)
- top predicted labels: rain (11), vacuum cleaner (10), clock tick (9), airplane (5), mouse click (4)
- top confusions: train -> rain (6), frog -> crow (3), mouse click -> keyboard typing (3), wind -> airplane (2), wind -> vacuum cleaner (2)
- hardest labels: train (8/8, 1.00), wind (8/8, 1.00), frog (6/8, 0.75), water drops (6/8, 0.75), crackling fire (5/8, 0.62)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/llm_predictions

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| train | 8 | 8 | 1.00 | rain (6) |
| wind | 8 | 8 | 1.00 | airplane (2) |
| frog | 6 | 8 | 0.75 | crow (3) |
| water drops | 6 | 8 | 0.75 | clock tick (2) |
| crackling fire | 5 | 8 | 0.62 | rain (2) |
| glass breaking | 5 | 8 | 0.62 | door wood creaks (1) |
| helicopter | 5 | 8 | 0.62 | airplane (2) |
| can opening | 4 | 8 | 0.50 | mouse click (2) |
| door wood creaks | 4 | 8 | 0.50 | chainsaw (1) |
| engine | 3 | 8 | 0.38 | vacuum cleaner (2) |

## Model: llm_predictions_guided
- errors: 99 / 400 (acc 0.7525)
- top true labels: wind (8), train (8), glass breaking (6), water drops (6), pig (5)
- top predicted labels: airplane (10), rain (9), mouse click (9), vacuum cleaner (7), door wood knock (6)
- top confusions: train -> rain (6), helicopter -> airplane (4), wind -> vacuum cleaner (3), keyboard typing -> mouse click (3), glass breaking -> door wood knock (3)
- hardest labels: train (8/8, 1.00), wind (8/8, 1.00), glass breaking (6/8, 0.75), water drops (6/8, 0.75), door wood creaks (5/8, 0.62)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/llm_predictions_guided

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| train | 8 | 8 | 1.00 | rain (6) |
| wind | 8 | 8 | 1.00 | vacuum cleaner (3) |
| glass breaking | 6 | 8 | 0.75 | door wood knock (3) |
| water drops | 6 | 8 | 0.75 | clock tick (2) |
| door wood creaks | 5 | 8 | 0.62 | chainsaw (1) |
| helicopter | 5 | 8 | 0.62 | airplane (4) |
| pig | 5 | 8 | 0.62 | rooster (2) |
| crackling fire | 4 | 8 | 0.50 | sea waves (1) |
| frog | 4 | 8 | 0.50 | crow (2) |
| keyboard typing | 4 | 8 | 0.50 | mouse click (3) |

## Model: panns_transfer
- errors: 38 / 400 (acc 0.9050)
- top true labels: pig (5), helicopter (5), frog (2), thunderstorm (2), crackling fire (2)
- top predicted labels: wind (5), washing machine (4), drinking sipping (3), airplane (3), crickets (3)
- top confusions: pig -> dog (2), frog -> crow (2), helicopter -> crickets (2), engine -> washing machine (2), airplane -> wind (2)
- hardest labels: helicopter (5/8, 0.62), pig (5/8, 0.62), airplane (2/8, 0.25), crackling fire (2/8, 0.25), engine (2/8, 0.25)
- audio copied to: outputs/results/run_20251221_003516/errors/audio/panns_transfer

| true_label | errors | total | error_rate | top_pred |
| --- | --- | --- | --- | --- |
| helicopter | 5 | 8 | 0.62 | crickets (2) |
| pig | 5 | 8 | 0.62 | dog (2) |
| airplane | 2 | 8 | 0.25 | wind (2) |
| crackling fire | 2 | 8 | 0.25 | rain (1) |
| engine | 2 | 8 | 0.25 | washing machine (2) |
| frog | 2 | 8 | 0.25 | crow (2) |
| thunderstorm | 2 | 8 | 0.25 | wind (1) |
| water drops | 2 | 8 | 0.25 | pouring water (1) |
| breathing | 1 | 8 | 0.12 | wind (1) |
| brushing teeth | 1 | 8 | 0.12 | hand saw (1) |
