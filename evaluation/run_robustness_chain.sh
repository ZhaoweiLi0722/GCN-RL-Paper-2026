#!/bin/zsh
# Robustness campaign chain: A top-up (umyo/fmyo) -> B (condition stress) -> C (forecast).
# Gated on Experiment A completing (rate_0.6 summary present). Every stage is resumable
# (per-CSV), so a sleep-kill loses only the in-flight seed: just relaunch this script.
#
#   caffeinate -i env PYTHONPATH=. ./evaluation/run_robustness_chain.sh >> chain.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# zsh does NOT word-split unquoted scalars, so keep the command as an array.
RUN=(/opt/anaconda3/bin/conda run -n gcn-rl --no-capture-output)
export CAMPAIGN_STEPS=${CAMPAIGN_STEPS:-150000}

echo "[chain] $(date) waiting for Experiment A to finish (rate_0.6 summary)"
until [ -f results/disruption_sweep/rate_0.6/summary.csv ]; do sleep 300; done

echo "[chain] $(date) A done -> topping up umyo/fmyo baselines (resumable)"
"${RUN[@]}" python -m evaluation.disruption_sweep

echo "[chain] $(date) -> Experiment B (condition stress)"
"${RUN[@]}" python -m evaluation.condition_stress

echo "[chain] $(date) -> Experiment C (forecast robustness, flagship)"
"${RUN[@]}" python -m evaluation.forecast_robustness

echo "[chain] $(date) ROBUSTNESS_CHAIN_DONE"
