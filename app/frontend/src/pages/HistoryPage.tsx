import { useEffect, useState } from "react";
import { listJobs, JobSummaryItem } from "../api/jobs";
import { formatBytes } from "../api/client";

type Props = {
  onSelectJob: (jobId: string) => void;
};

const STATUS_LABEL: Record<string, string> = {
  queued: "Chờ xử lý",
  running: "Đang xử lý",
  done: "Hoàn thành",
  error: "Lỗi",
};

const STATUS_COLOR: Record<string, string> = {
  queued: "#b08a00",
  running: "#1a6fa8",
  done: "#1a7a4a",
  error: "#c0392b",
};

export default function HistoryPage({ onSelectJob }: Props) {
  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listJobs().then((list) => {
      setJobs(list);
      setLoading(false);
    });
  }, []);

  return (
    <main className="page">
      <div className="history-panel">
        <div className="section-title">
          <span>7 video gần nhất</span>
          <strong>Lịch sử phân tích</strong>
        </div>

        {loading && <p className="history-empty">Đang tải...</p>}

        {!loading && jobs.length === 0 && (
          <p className="history-empty">Chưa có video nào được phân tích.</p>
        )}

        {!loading && jobs.length > 0 && (
          <ul className="history-list">
            {jobs.map((job) => (
              <li key={job.id} className="history-item">
                <div className="history-info">
                  <span className="history-name">{job.input.original_name}</span>
                  <span className="history-meta">
                    {formatBytes(job.input.size_bytes)}
                    {job.processing.model_name && ` · ${job.processing.model_name.replace(".pth", "")}`}
                    {job.summary?.scene_count != null && ` · ${job.summary.scene_count} scene`}
                  </span>
                  <span className="history-date">
                    {new Date(job.created_at).toLocaleString("vi-VN")}
                  </span>
                </div>
                <div className="history-right">
                  <span className="history-status" style={{ color: STATUS_COLOR[job.status] ?? "#555" }}>
                    {STATUS_LABEL[job.status] ?? job.status}
                  </span>
                  {job.status === "done" && (
                    <button className="history-view-btn" onClick={() => onSelectJob(job.id)}>
                      Xem kết quả
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}
