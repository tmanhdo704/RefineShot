import { Activity, Clock, Cpu, Film } from "lucide-react";

import { RefineShotJob, formatTime } from "../api/client";

type Props = {
  job: RefineShotJob;
};

export default function SummaryPanel({ job }: Props) {
  const summary = job.summary;
  const processing = job.processing;
  const items = [
    { label: "Scenes", value: summary?.scene_count ?? 0, icon: Film },
    { label: "Cuts", value: summary?.boundary_count ?? 0, icon: Activity },
    { label: "Average", value: formatTime(summary?.average_scene_duration_sec ?? 0), icon: Clock },
    { label: "Longest", value: formatTime(summary?.longest_scene_sec ?? 0), icon: Clock }
  ];

  const processingMeta = [
    processing.model,
    processing.device ? `device: ${processing.device}` : null,
    processing.threshold !== undefined ? `thr: ${processing.threshold}` : null,
    processing.temperature !== undefined && processing.temperature !== null ? `T: ${processing.temperature}` : null,
    processing.sigma !== undefined && processing.sigma !== null ? `σ: ${processing.sigma}` : null,
  ].filter(Boolean);

  return (
    <>
      <section className="summary-grid">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <div className="stat" key={item.label}>
              <Icon size={18} />
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          );
        })}
      </section>
      {processingMeta.length > 0 && (
        <section className="processing-meta">
          <Cpu size={14} />
          <span>{processingMeta.join(" · ")}</span>
        </section>
      )}
    </>
  );
}
