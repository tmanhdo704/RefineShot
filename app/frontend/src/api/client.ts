export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type Boundary = {
  index: number;
  frame: number;
  time_sec: number;
  confidence: number;
  type: string;
};

export type Scene = {
  index: number;
  start_frame: number;
  end_frame: number;
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  thumbnail_url?: string;
};

export type JobSummary = {
  boundary_count: number;
  scene_count: number;
  average_scene_duration_sec: number;
  shortest_scene_sec: number;
  longest_scene_sec: number;
};

export type RefineShotJob = {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  created_at: string;
  expires_at: string;
  input: {
    original_name: string;
    size_bytes: number;
    content_type: string;
    fps?: number;
    duration_sec?: number;
    total_frames?: number;
    width?: number;
    height?: number;
  };
  processing: {
    model: string;
    display_name?: string;
    preset?: string;
    device?: string;
    backend?: string;
    threshold?: number;
    temperature?: number;
    sigma?: number;
    sensitivity?: string;
    min_scene_duration_sec?: number;
    checkpoint?: string;
  };
  storage?: {
    video_url?: string;
  };
  summary?: JobSummary | null;
  boundaries: Boundary[];
  scenes: Scene[];
  exports?: {
    json_url?: string;
    csv_url?: string;
    txt_url?: string;
  };
  artifacts?: {
    storyboard_url?: string | null;
  };
  error?: string | null;
};

export function mediaUrl(url?: string | null): string {
  if (!url) return "";
  if (url.startsWith("http")) return url;
  return `${API_BASE}${url}`;
}

export function formatTime(seconds = 0): string {
  const safe = Math.max(0, seconds);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.floor((safe % 1) * 1000);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}

export function formatBytes(bytes = 0): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
