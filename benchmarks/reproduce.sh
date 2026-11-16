#!/bin/bash
# Regenerates every number in docs/BENCHMARKS.md. Run from the repository root:
#     bash benchmarks/reproduce.sh          # rule-based results (seconds)
#     bash benchmarks/reproduce.sh --nli    # also the NLI ablations (downloads a ~330 MB model)
set -e
cd "$(dirname "$0")"
R=results
mkdir -p $R

for split in dev test; do
  python run_eval.py --system baseline --split $split --out $R/baseline_$split.json
  python run_eval.py --system current  --split $split --errors --out $R/current_$split.json
done

# Ablations on the held-out split
python run_eval.py --system current --split test --verifiers overclaim   --out $R/ablation_overclaim_only.json
python run_eval.py --system current --split test --verifiers constraints --out $R/ablation_constraints_only.json
python run_eval.py --system current --split test --no-negation           --out $R/ablation_no_negation.json

python retrieval_eval.py | tee $R/retrieval.txt

python perf.py --system baseline --out $R/perf_baseline.json
python perf.py --system current  --out $R/perf_current.json

# Real prose (needs network access to en.wikipedia.org)
python real_text_flags.py --system baseline --set tuning  --out $R/real_text_tuning_baseline.json
python real_text_flags.py --system current  --set tuning  --out $R/real_text_tuning_current.json
python real_text_flags.py --system baseline --set heldout --out $R/real_text_heldout_baseline.json
python real_text_flags.py --system current  --set heldout --out $R/real_text_heldout_current.json

if [ "$1" = "--nli" ]; then
  python nli_sweep.py --split dev | tee $R/nli_sweep_dev.txt
  python run_eval.py --system current --split test --nli --out $R/ablation_nli.json
  python run_eval.py --system current --split test --nli --nli-contradiction 0.95 --errors --out $R/ablation_nli_contradiction.json
  python run_eval.py --system current --split test --verifiers "" --nli --out $R/ablation_nli_only.json
fi
