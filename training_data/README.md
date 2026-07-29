# RTX 4090 Bootstrap Data

`regional_4090_training_data.zip` contains six synthetic teacher-cache files
needed by the matched regional-drift GCN/flat residual DDPG experiment.

SHA-256:

```text
f7792fa97a889f463f22ebae2d2846cc475225412141a1f377b4a84d55efec17
```

The caches contain simulation states, residual action labels, advantages, and
scenario metadata. They contain no patient records or personal data.

Use `scripts/prepare_regional_training_data.ps1` to verify and extract them.
The extracted `results/` files remain ignored by Git.
