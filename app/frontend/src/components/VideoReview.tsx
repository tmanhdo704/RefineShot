import { useRef, useState } from "react";

import { RefineShotJob, formatTime, mediaUrl } from "../api/client";
import BoundaryTimeline from "./BoundaryTimeline";

type Props = {
  job: RefineShotJob;
  onSeekReady: (seek: (time: number) => void) => void;
};

export default function VideoReview({ job, onSeekReady }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const duration = job.input.duration_sec || videoRef.current?.duration || 0;

  const seek = (time: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = time;
    videoRef.current.play().catch(() => undefined);
  };

  onSeekReady(seek);

  return (
    <section className="review-layout">
      <div className="video-frame">
        <video
          ref={videoRef}
          controls
          src={mediaUrl(job.storage?.video_url)}
          onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          onLoadedMetadata={(event) => setCurrentTime(event.currentTarget.currentTime)}
        />
      </div>
      <BoundaryTimeline
        duration={duration}
        currentTime={currentTime}
        boundaries={job.boundaries || []}
        onSeek={seek}
      />
      <div className="time-row">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </section>
  );
}
