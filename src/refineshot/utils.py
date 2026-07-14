import subprocess
from collections.abc import Iterator

import numpy as np

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

# predictions_to_scenes / evaluate_scenes live canonically in refineshot.eval; re-use them here
# instead of keeping a second copy of the matching algorithm (which risked silent divergence).
from refineshot.eval import (
    evaluate_scenes as _evaluate_scenes_core,
)
from refineshot.eval import (
    predictions_to_scenes,
)


def get_frames(fn: str, width: int = 48, height: int = 27) -> np.ndarray:
    if ffmpeg is not None:
        video_stream, err = (
            ffmpeg
            .input(fn)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
    else:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", fn,
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        video_stream = proc.stdout
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video

def get_batches(frames: np.ndarray) -> Iterator[np.ndarray]:
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i + 100]

    return func()

def scenes2zero_one_representation(scenes: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    prev_end = 0
    one_hot = np.zeros([n_frames], np.uint64)
    many_hot = np.zeros([n_frames], np.uint64)

    for start, end in scenes:
        # number of frames in transition: start - prev_end - 1 (hardcut has 0)

        # values of many_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0][1] -> 0
        # [0][x][2] -> 0, 1
        # [0][x][x][3] -> 0, 1, 2
        # [0][x][x][x][4] -> 0, 1, 2, 3
        # [0][x][x][x][x][5] -> 0, 1, 2, 3, 4
        for i in range(prev_end, start):
            many_hot[i] = 1

        # values of one_hot_index
        # frame with index (0..n-1) is from a scene, frame [x] is a transition frame
        # [0]|[1] -> 0
        # [0][x]|[2] -> 1
        # [0][x]|[x][3] -> 1
        # [0][x][x]|[x][4] -> 2
        # [0][x][x]|[x][x][5] -> 2
        # ...
        if not (prev_end == 0 and start == 0):
            one_hot_index = prev_end + (start - prev_end) // 2
            one_hot[one_hot_index] = 1

        prev_end = end

    # if scene ends with transition
    if prev_end + 1 != n_frames:
        for i in range(prev_end, n_frames):
            many_hot[i] = 1

        one_hot_index = prev_end + (n_frames - prev_end) // 2
        one_hot[one_hot_index] = 1

    return one_hot, many_hot

def evaluate_scenes(
    gt_scenes: np.ndarray,
    pred_scenes: np.ndarray,
    return_mistakes: bool = False,
    n_frames_miss_tolerance: int = 2,
) -> tuple:
    """Precision/recall/F1 (+ tp/fp/fn) for scene boundaries.

    Thin wrapper over :func:`refineshot.eval.evaluate_scenes` (the single source of the
    two-pointer matching algorithm) that adds precision/recall/F1 for the training-side
    callers. ``predictions_to_scenes`` is likewise re-exported from there.
    """
    if return_mistakes:
        tp, fp, fn, fp_mistakes, fn_mistakes = _evaluate_scenes_core(
            gt_scenes, pred_scenes, tolerance=n_frames_miss_tolerance, return_mistakes=True
        )
    else:
        tp, fp, fn = _evaluate_scenes_core(gt_scenes, pred_scenes, tolerance=n_frames_miss_tolerance)

    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    # NOT common.f1_pr: (p * r * 2) has a different float operation order than
    # 2 * p * r and can differ in the last bit; historical numbers are pinned to it.
    f1 = (p * r * 2) / (p + r) if p + r != 0 else 0

    if return_mistakes:
        return p, r, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return p, r, f1, (tp, fp, fn)

def mAP_f1_p_fix_r(
    one_hot_pred: dict[str, np.ndarray],
    gt_scenes: dict[str, np.ndarray],
    fixed_r: float = 0.70654,
    skip_map_miou: bool = True,
) -> tuple:
    if fixed_r > 0:
        assert skip_map_miou
        eps = 0.001
        l_thr = 0.
        h_thr = 1.
        while h_thr - l_thr > eps:
            cur_thr = (l_thr + h_thr) / 2.
            precision = recall = f1 = tp = fp = fn = 0
            for file_name, pred in one_hot_pred.items():
                pred_scenes = predictions_to_scenes((pred > np.array([cur_thr])).astype(np.uint8))
                _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
                tp += tp_
                fp += fp_
                fn += fn_

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp * 1. / (tp + fp)
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp * 1. / (tp + fn)

            if recall > fixed_r + eps:
                l_thr = cur_thr
            elif recall < fixed_r - eps:
                h_thr = cur_thr
            else:
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = (precision * recall * 2) / (precision + recall)
                return 0, f1, precision, recall, cur_thr, 0
        precision = recall = f1 = tp = fp = fn = 0
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > np.array([l_thr])).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp += tp_
            fp += fp_
            fn += fn_

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp * 1. / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp * 1. / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (precision * recall * 2) / (precision + recall)
        return 0, f1, precision, recall, cur_thr, 0

    # f1 p r threshold
    thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.2833, 0.2867, 0.29, 0.292, 0.294, 0.296, 0.298, 0.3, 0.302, 0.304, 0.306, 0.308, 0.31, 0.3133, 0.3167, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.5, 0.6, 0.7, 0.8,  # noqa: E501
                           0.9])
#     thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.294, 0.2945, 0.295, 0.2952, 0.2954, 0.2956, 0.2958, 0.296, 0.2962, 0.2964, 0.2966, 0.2968, 0.297, 0.2975, 0.298, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # noqa: E501
#                            0.9])
#     thresholds = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
#                            0.9])
    precision, recall, f1, tp, fp, fn = np.zeros_like(thresholds), np.zeros_like(thresholds), \
                                        np.zeros_like(thresholds), np.zeros_like(thresholds), \
                                        np.zeros_like(thresholds), np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        for file_name, pred in one_hot_pred.items():
            pred_scenes = predictions_to_scenes((pred > thresholds[i]).astype(np.uint8))
            _, _, _, (tp_, fp_, fn_) = evaluate_scenes(gt_scenes[file_name], pred_scenes)
            tp[i] += tp_
            fp[i] += fp_
            fn[i] += fn_

        if tp[i] + fp[i] == 0:
            precision[i] = 0
        else:
            precision[i] = tp[i] * 1. / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall[i] = 0
        else:
            recall[i] = tp[i] * 1. / (tp[i] + fn[i])
        if precision[i] + recall[i] == 0:
            f1[i] = 0
        else:
            f1[i] = (precision[i] * recall[i] * 2) / (precision[i] + recall[i])

    best_idx = np.argmax(f1)

    if skip_map_miou:
        return 0, f1[best_idx], precision[best_idx], recall[best_idx], thresholds[best_idx], 0

    # The upstream AutoShot mAP/mIOU branch depended on evaluate_scenes_mAP,
    # cal_miou and sklearn's average_precision_score, none of which were ported;
    # calling it would NameError. All callers in this repo use skip_map_miou=True.
    raise NotImplementedError("mAP/mIOU evaluation was not ported from upstream AutoShot; use skip_map_miou=True")
