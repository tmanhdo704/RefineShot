# Data layout

Datasets and generated caches stay local because of their size and licensing.
Use the following layout when training or reproducing cached evaluation:

```text
data/
|-- shot/
|-- clipshots/
|-- bbc/
|-- shot_clipshots_trainval.pickle
|-- shot_clipshots_phase2_sample_cache.pkl
`-- eval/
    |-- gt_scenes_dict_baseline_v2.pickle
    |-- shot_gt200_logits_cv2.pkl
    `-- additional generated logits and caches
```

The four model checkpoints are versioned separately under
`models/checkpoints/` with Git LFS. Training outputs and resumable state are
written under `runs/` and are ignored by Git.
