## Why this fork exists

This is a fork of the official DiffRegCD implementation
(https://github.com/Anita-Madani/DiffRegCD-Integrated-Registration-and-Change-Detection-with-Diffusion-Features),
with one bug fixed that causes the released code to crash during
evaluation, plus one data-preparation utility added for a commonly
circulated repackaging of LEVIR-CD256 that isn't in the layout the
authors' own data loader expects natively.

All credit for the architecture, training code, and released weights
belongs to the original authors (Madani, Chellappa & Patel, WACV 2026).
This fork exists purely to make reproduction easier for others working
from the same starting point we did.

### 1. Bug fix — `model/cd_model_256.py`

`test()` assigns the registration model's output to
`self.pred_corase_flow` (typo: "corase", not "coarse") — but every
other method that reads this value, including `get_current_visuals()`,
expects it under `self.pred_flow_coarse`. Because the two names never
match, `test()` runs to completion and logs all quantitative metrics
correctly, then crashes immediately afterward with:

```
AttributeError: 'CD' object has no attribute 'pred_flow_coarse'
```

**This does not affect the reported metrics.** mF1 / mIoU / OA and
per-class precision/recall are all computed and logged *before* the
crash — the failure is strictly in the downstream visualization step
(`get_current_visuals()`, called to save prediction images), which
never runs. If you only need the quantitative numbers, the bug is
otherwise silent; it only surfaces when the test loop tries to render
output images.

**The fix** — two occurrences, one per branch of the
`isinstance(self.netCD, nn.DataParallel)` check inside `test()`:

```diff
- self.pred_corase_flow, self.logits = self.netReg(fA, fB)
+ self.pred_flow_coarse, self.logits = self.netReg(fA, fB)
```

A rename to match the attribute name already used consistently
everywhere else in the file.

### 2. Data preparation — `make_dataset_layout_flat.py` (new file, not from upstream)

The authors' `CMUCDFlowDataset` expects a specific directory layout
(`srcA/`, `train/t0/`, `train/mask/`, `list/*.txt`). A commonly
circulated repackaging of LEVIR-CD256 instead ships as flat `A/`,
`B/`, `label/` folders with the train/val/test split encoded as a
filename prefix (e.g. `train_1_1.png`, `test_100_1.png`). This is
**not a bug in DiffRegCD** — it's a mismatch between that particular
data repackaging and the layout the authors' own loader was built
around. `make_dataset_layout_flat.py` bridges the two: it reads the
flat, prefix-split format and writes out the directory structure the
dataset class expects, then hands off to the authors' own
`scripts/prepare_gt_flow.py` for the synthetic-affine-perturbation /
ground-truth-flow generation step.

If your LEVIR-CD256 download already matches the expected layout
natively, this script isn't needed at all.

### Verification

Reproduced independently across multiple environment rebuilds, on
LEVIR-CD256, using the official released weights unmodified:

| | mF1 | mIoU | OA |
|---|---|---|---|
| Paper-reported | 0.929 | 0.881 | 0.987 |
| This fork, run 1 | 0.939 | 0.891 | 0.989 |
| This fork, run 2 | 0.9395 | 0.8909 | 0.9886 |

Consistently above the paper's own reported numbers, using a different
affine-perturbation seed than the original authors' — expected given
the shared evaluation protocol, and confirms the patch changes nothing
about the model's actual behavior, only whether the test script
finishes without crashing.

### Limitations of this verification, and the planned next step

The results above confirm the pipeline reproduces the paper's own
numbers correctly — they do not by themselves confirm robustness
beyond what the paper already tested. Specifically:

- **Only LEVIR-CD has been run in this fork.** The paper also reports
  results on WHU-CD, DSIFN-CD, SYSU-CD, and VL-CMU-CD; those have not
  been independently reproduced here.
- **Only synthetic affine perturbation has been tested** — random
  translation, rotation, and scale, matching the paper's own
  evaluation protocol. This does not capture true parallax-induced
  misalignment, which is terrain-dependent (displacement varies with
  object height, not uniform across the scene) and arises from
  genuine differences in sensor look/squint angle — the actual
  condition this pipeline would need to handle for cross-vendor
  commercial imagery lacking DSM/DTM correction.
- **Only building change detection has been verified in this fork's
  own run.** The paper's broader benchmark set includes some
  non-building land-cover change (SYSU-CD, VL-CMU-CD), but that
  hasn't been reproduced here either, and disaster-specific change
  types (flooding, etc.) were not part of the original paper's
  training or evaluation data at all.

**Planned next step**: validating registration robustness under real
(not synthetic) viewing-angle-driven parallax, using SpaceNet MVOI —
27 same-day, same-scene collects of Atlanta at off-nadir angles from
7° to 54°, which lets true parallax be isolated as a null-hypothesis
test (since nothing in the scene actually changed between collects,
any predicted "change" is by definition a registration artifact, not
a real detection). Not yet included in this fork; noted here as the
intended direction, not a completed result.

### If you're working from upstream instead of this fork

The bug fix doesn't persist across a fresh `git clone` of the original
repository. If you clone upstream directly rather than using this
fork, you'll need to reapply the one-line patch above before `-p test`
will run to completion.