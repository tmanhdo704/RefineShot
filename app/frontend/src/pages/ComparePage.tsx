import { FormEvent, useEffect, useRef, useState } from "react";
import {
  createCompareFromUpload,
  getJob,
  listModels,
  methodDisplayName,
  modelDisplayName,
  ModelInfo,
} from "../api/jobs";
import { RefineShotJob, Boundary, mediaUrl, formatTime } from "../api/client";
import UploadDropzone from "../components/UploadDropzone";
import ErrorState from "../components/ErrorState";

const COLOR_A = "#2563eb";
const COLOR_B = "#ea580c";
const OVERLAP_THRESHOLD_SEC = 0.5;

type CompareState =
  | { phase: "form" }
  | { phase: "processing"; jobA: RefineShotJob; jobB: RefineShotJob }
  | { phase: "done"; jobA: RefineShotJob; jobB: RefineShotJob };

export default function ComparePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [presetA, setPresetA] = useState("best_shot");
  const [presetB, setPresetB] = useState("best_clipshot");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [state, setState] = useState<CompareState>({ phase: "form" });
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listModels().then((list) => {
      if (list.length === 0) return;
      setModels(list.filter((m) => m.available));
    });
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const startPolling = (jobA: RefineShotJob, jobB: RefineShotJob) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const [a, b] = await Promise.all([getJob(jobA.id), getJob(jobB.id)]);
      if ((a.status === "done" || a.status === "error") &&
          (b.status === "done" || b.status === "error")) {
        clearInterval(pollRef.current!);
        setState({ phase: "done", jobA: a, jobB: b });
      } else {
        setState({ phase: "processing", jobA: a, jobB: b });
      }
    }, 1500);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) { setError("Chọn file video trước."); return; }
    if (presetA === presetB) { setError("Chọn 2 model khác nhau."); return; }
    setError(null);
    setIsSubmitting(true);
    try {
      const { job_a, job_b } = await createCompareFromUpload(file, presetA, presetB);
      setState({ phase: "processing", jobA: job_a, jobB: job_b });
      startPolling(job_a, job_b);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload thất bại.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const seekTo = (sec: number) => {
    if (videoRef.current) videoRef.current.currentTime = sec;
  };

  if (state.phase === "form") {
    return (
      <main className="page upload-page">
        <form className="upload-panel" onSubmit={handleSubmit}>
          <div className="section-title">
            <span>So sánh</span>
            <strong>Chạy 2 model trên cùng 1 video</strong>
          </div>

          <UploadDropzone file={file} disabled={isSubmitting} onFileChange={setFile} />

          {models.length > 0 && (
            <div className="settings-row compare-model-row">
              <label>
                <span style={{ color: COLOR_A }}>■ Model A</span>
                <select value={presetA} disabled={isSubmitting} onChange={(e) => setPresetA(e.target.value)}>
                  {models.map((m) => <option key={m.preset} value={m.preset}>{modelDisplayName(m)}</option>)}
                </select>
              </label>
              <label>
                <span style={{ color: COLOR_B }}>■ Model B</span>
                <select value={presetB} disabled={isSubmitting} onChange={(e) => setPresetB(e.target.value)}>
                  {models.map((m) => <option key={m.preset} value={m.preset}>{modelDisplayName(m)}</option>)}
                </select>
              </label>
            </div>
          )}

          {error && <ErrorState message={error} />}

          <button className="primary-action" type="submit" disabled={isSubmitting || !file}>
            {isSubmitting ? "Đang tải lên..." : "So sánh"}
          </button>
        </form>
      </main>
    );
  }

  const { jobA, jobB } = state;
  const doneA = jobA.status === "done";
  const doneB = jobB.status === "done";
  const bothDone = doneA && doneB;
  const duration = jobA.input.duration_sec ?? jobB.input.duration_sec ?? 0;

  const boundariesA: Boundary[] = doneA ? (jobA.boundaries ?? []) : [];
  const boundariesB: Boundary[] = doneB ? (jobB.boundaries ?? []) : [];

  // Detect overlapping boundaries
  const overlapSet = new Set<number>();
  boundariesA.forEach((a) => {
    if (boundariesB.some((b) => Math.abs(a.time_sec - b.time_sec) <= OVERLAP_THRESHOLD_SEC)) {
      overlapSet.add(a.index);
    }
  });

  const videoUrl = mediaUrl(jobA.storage?.video_url);

  return (
    <main className="page compare-page">
      <div className="compare-panel">
        <div className="section-title">
          <span>Kết quả so sánh</span>
          <strong>{jobA.input.original_name}</strong>
        </div>

        {/* Progress */}
        {!bothDone && (
          <div className="compare-progress-row">
            <div className="compare-progress-item">
              <span style={{ color: COLOR_A }}>■ {methodDisplayName(jobA.processing.display_name) ?? "Model A"}</span>
              <span>{Math.round((jobA.progress ?? 0) * 100)}% — {jobA.stage}</span>
            </div>
            <div className="compare-progress-item">
              <span style={{ color: COLOR_B }}>■ {methodDisplayName(jobB.processing.display_name) ?? "Model B"}</span>
              <span>{Math.round((jobB.progress ?? 0) * 100)}% — {jobB.stage}</span>
            </div>
          </div>
        )}

        {/* Video player */}
        {videoUrl && (
          <video ref={videoRef} className="compare-video" src={videoUrl} controls />
        )}

        {/* Legend */}
        {bothDone && (
          <div className="compare-legend">
            <span style={{ color: COLOR_A }}>■ {methodDisplayName(jobA.processing.display_name) ?? "Model A"} ({boundariesA.length} boundary)</span>
            <span style={{ color: COLOR_B }}>■ {methodDisplayName(jobB.processing.display_name) ?? "Model B"} ({boundariesB.length} boundary)</span>
            {overlapSet.size > 0 && (
              <span className="compare-overlap-note">⬡ {overlapSet.size} boundary trùng nhau (± {OVERLAP_THRESHOLD_SEC}s)</span>
            )}
          </div>
        )}

        {/* Timeline */}
        {bothDone && duration > 0 && (
          <div className="compare-timeline">
            {boundariesA.map((b) => (
              <button
                key={`a-${b.index}`}
                className="compare-marker"
                style={{
                  left: `${(b.time_sec / duration) * 100}%`,
                  background: overlapSet.has(b.index) ? "#7c3aed" : COLOR_A,
                  zIndex: overlapSet.has(b.index) ? 3 : 2,
                }}
                title={`Model A · ${formatTime(b.time_sec)}`}
                onClick={() => seekTo(b.time_sec)}
              />
            ))}
            {boundariesB.map((b) => (
              <button
                key={`b-${b.index}`}
                className="compare-marker compare-marker-b"
                style={{
                  left: `${(b.time_sec / duration) * 100}%`,
                  background: COLOR_B,
                }}
                title={`Model B · ${formatTime(b.time_sec)}`}
                onClick={() => seekTo(b.time_sec)}
              />
            ))}
          </div>
        )}

        {/* Scene columns */}
        {bothDone && (
          <div className="compare-scenes-grid">
            <div className="compare-scenes-col">
              <h3 style={{ color: COLOR_A }}>■ {methodDisplayName(jobA.processing.display_name) ?? "Model A"} — {jobA.summary?.scene_count ?? 0} scene</h3>
              <ul className="compare-scene-list">
                {(jobA.scenes ?? []).map((s) => (
                  <li key={s.index} className="compare-scene-item" onClick={() => seekTo(s.start_time_sec)}>
                    {s.thumbnail_url && <img src={mediaUrl(s.thumbnail_url)} alt="" className="compare-thumb" />}
                    <span className="compare-scene-time">{formatTime(s.start_time_sec)} – {formatTime(s.end_time_sec)}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="compare-scenes-col">
              <h3 style={{ color: COLOR_B }}>■ {methodDisplayName(jobB.processing.display_name) ?? "Model B"} — {jobB.summary?.scene_count ?? 0} scene</h3>
              <ul className="compare-scene-list">
                {(jobB.scenes ?? []).map((s) => (
                  <li key={s.index} className="compare-scene-item" onClick={() => seekTo(s.start_time_sec)}>
                    {s.thumbnail_url && <img src={mediaUrl(s.thumbnail_url)} alt="" className="compare-thumb" />}
                    <span className="compare-scene-time">{formatTime(s.start_time_sec)} – {formatTime(s.end_time_sec)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
