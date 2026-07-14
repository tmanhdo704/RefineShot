import { ArrowLeft, RefreshCcw, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { RefineShotJob, mediaUrl } from "../api/client";
import { deleteJob, getJob } from "../api/jobs";
import ErrorState from "../components/ErrorState";
import ExportPanel from "../components/ExportPanel";
import JobProgress from "../components/JobProgress";
import SceneList from "../components/SceneList";
import SummaryPanel from "../components/SummaryPanel";
import VideoReview from "../components/VideoReview";

type Props = {
  jobId: string;
  onBack: () => void;
};

export default function ResultPage({ jobId, onBack }: Props) {
  const [job, setJob] = useState<RefineShotJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const seekRef = useRef<(time: number) => void>(() => undefined);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const load = async () => {
      try {
        const nextJob = await getJob(jobId);
        if (!active) return;
        setJob(nextJob);
        setError(null);
        if (nextJob.status !== "done" && nextJob.status !== "error") {
          timer = window.setTimeout(load, 1400);
        }
      } catch (exc) {
        if (!active) return;
        setError(exc instanceof Error ? exc.message : "Cannot load this job.");
      }
    };

    load();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId]);

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await deleteJob(jobId);
      onBack();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Delete failed.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <main className="page result-page">
      <div className="result-header">
        <button className="ghost-button" type="button" onClick={onBack}>
          <ArrowLeft size={18} />
          New
        </button>
        <div>
          <span className="eyebrow">{job?.status || "loading"}</span>
          <h1>{job?.input.original_name || jobId}</h1>
        </div>
        <button className="icon-button" type="button" title="Delete job" disabled={isDeleting} onClick={handleDelete}>
          <Trash2 size={18} />
        </button>
      </div>

      {error && <ErrorState message={error} />}

      {!job && !error && (
        <section className="loading-panel">
          <RefreshCcw size={20} />
          Loading job
        </section>
      )}

      {job && (
        <>
          <JobProgress status={job.status} stage={job.stage} progress={job.progress} />

          {job.status === "error" && <ErrorState title="Analysis failed" message={job.error || "Unknown error."} />}

          {job.status === "done" && (
            <>
              <SummaryPanel job={job} />
              <VideoReview job={job} onSeekReady={(seek) => (seekRef.current = seek)} />
              <div className="result-grid">
                <SceneList job={job} onSeek={(time) => seekRef.current(time)} />
                <div className="result-sidebar">
                  <ExportPanel job={job} />
                  {job.artifacts?.storyboard_url && (
                    <section className="panel storyboard-panel">
                      <div className="panel-heading">
                        <span>Storyboard</span>
                      </div>
                      <a href={mediaUrl(job.artifacts.storyboard_url)} target="_blank" rel="noreferrer">
                        <img src={mediaUrl(job.artifacts.storyboard_url)} alt="" />
                      </a>
                    </section>
                  )}
                </div>
              </div>
            </>
          )}
        </>
      )}
    </main>
  );
}
