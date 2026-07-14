import { Boundary } from "../api/client";

type Props = {
  duration: number;
  currentTime: number;
  boundaries: Boundary[];
  onSeek: (time: number) => void;
};

export default function BoundaryTimeline({ duration, currentTime, boundaries, onSeek }: Props) {
  const safeDuration = Math.max(0.001, duration || 0);
  const playheadLeft = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

  return (
    <div className="timeline" aria-label="Boundary timeline">
      <div className="timeline-base" />
      {boundaries.map((boundary) => {
        const left = Math.min(100, Math.max(0, (boundary.time_sec / safeDuration) * 100));
        return (
          <button
            key={`${boundary.index}-${boundary.frame}`}
            className="boundary-marker"
            style={{ left: `${left}%` }}
            title={`${boundary.time_sec.toFixed(3)}s`}
            type="button"
            onClick={() => onSeek(boundary.time_sec)}
          >
            <span />
          </button>
        );
      })}
      <div className="playhead" style={{ left: `${playheadLeft}%` }} />
    </div>
  );
}
