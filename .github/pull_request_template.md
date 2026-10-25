## What and why

<!-- One paragraph: the change and the reason for it. Link the issue. -->

## Evidence

- [ ] Tests added or updated (`pytest -q` passes)
- [ ] `ruff check flowtrace tests` and `mypy` pass
- [ ] Numerics changed? MMS order unchanged (`flowtrace benchmark --only mms_mac,mms_simplec_quads,mms_simplec_tri`)
- [ ] A published number changed? `flowtrace benchmark --tier full` rerun and README/validation report updated
- [ ] New compiled/CUDA code? Python fallback kept and an equality test added
- [ ] Docs updated (`docs/`), CHANGELOG entry added

## Numbers before / after (if relevant)

| Quantity | Before | After | Reference |
|---|---|---|---|
