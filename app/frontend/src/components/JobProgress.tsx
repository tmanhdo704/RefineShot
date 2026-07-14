type Props = {
  status: string;
  stage: string;
  progress: number;
};

export default function JobProgress({ status, stage, progress }: Props) {
  const percent = Math.round(Math.max(0, Math.min(1, progress || 0)) * 100);

  return (
    <section className="panel progress-panel">
      <div className="panel-heading">
        <span>{status}</span>
        <strong>{percent}%</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <div className="stage-line">{stage.replace(/_/g, " ")}</div>
    </section>
  );
}
