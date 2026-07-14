import { Play } from "lucide-react";

import { RefineShotJob, formatTime, mediaUrl } from "../api/client";

type Props = {
  job: RefineShotJob;
  onSeek: (time: number) => void;
};

export default function SceneList({ job, onSeek }: Props) {
  return (
    <section className="panel scene-panel">
      <div className="panel-heading">
        <span>Scenes</span>
        <strong>{job.scenes?.length || 0}</strong>
      </div>
      <div className="scene-grid">
        {(job.scenes || []).map((scene) => (
          <button className="scene-item" type="button" key={scene.index} onClick={() => onSeek(scene.start_time_sec)}>
            <img src={mediaUrl(scene.thumbnail_url)} alt="" />
            <span className="scene-play">
              <Play size={14} />
            </span>
            <span className="scene-meta">
              <strong>Scene {scene.index + 1}</strong>
              <small>
                {formatTime(scene.start_time_sec)} - {formatTime(scene.end_time_sec)}
              </small>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
