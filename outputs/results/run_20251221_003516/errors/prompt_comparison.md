# Prompt Comparison

- base: `outputs/results/run_20251221_003516/predictions/llm_predictions.csv`
- compare: `outputs/results/run_20251221_003516/predictions/llm_predictions_guided.csv`
- common samples: 400
- base accuracy: 312/400 (0.7800)
- compare accuracy: 301/400 (0.7525)
- changed predictions: 83
- base-only correct: 31
- compare-only correct: 20

## Base-only correct (top true labels)
keyboard typing (4), hen (3), pig (2), thunderstorm (2), helicopter (2), vacuum cleaner (2), glass breaking (2), footsteps (2), fireworks (1), door wood creaks (1)

## Compare-only correct (top true labels)
helicopter (2), frog (2), clock tick (2), snoring (2), hen (2), crackling fire (1), rain (1), insects (1), engine (1), crickets (1)

## Most frequent changes (true -> base -> compare)
- keyboard typing -> keyboard typing -> mouse click (3)
- pig -> pig -> dog (2)
- thunderstorm -> thunderstorm -> fireworks (2)
- helicopter -> helicopter -> airplane (2)
- snoring -> breathing -> snoring (2)
- glass breaking -> glass breaking -> door wood knock (2)
- pig -> laughing -> rooster (1)
- pig -> crying baby -> rooster (1)
- pig -> coughing -> crow (1)
- frog -> crow -> rooster (1)
