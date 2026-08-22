# Recovery 1: Multi-Scenario DDPG Attribution Pilot

## Trigger

The original locked pilot exited during GCN seed 0 advantage-filtered
distillation pretraining at epoch 195 of 300. It completed zero online episodes
and produced no checkpoint or training manifest. The failed output and launcher
logs remain untouched.

## Recovery contract

Recovery 1 preserves the original scientific protocol exactly:

- the same graph and parameter-matched flat algorithms;
- the same teacher assets and SHA256 checks;
- the same seeds, 300 pretraining epochs, and 100 online episodes;
- the same four scenarios and round-robin schedule;
- the same model, optimizer, exploration, action, and safety settings;
- the same final and frozen-pretrain CRN evaluation settings;
- the same holdout seed and bootstrap comparison.

Only execution-level metadata changes:

- all three output roots use the suffix `recovery1`;
- Python fault handling and PyTorch C++ stack traces are enabled;
- the raw training-process exit code is written to the transcript;
- the final archive and provenance identify the recovery attempt.

The recovery runner refuses to overwrite an existing recovery output. It must
be launched at most once after confirming that the original process tree has
ended and the failed evidence remains intact.
