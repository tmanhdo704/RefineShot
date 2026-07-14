import { API_BASE, RefineShotJob } from "./client";

export type ModelInfo = {
  preset: string;
  display_name: string;
  is_default: boolean;
  available: boolean;
};

export function methodDisplayName(displayName: string | undefined): string | undefined {
  return displayName;
}

export function modelDisplayName(model: ModelInfo): string {
  return methodDisplayName(model.display_name) ?? model.display_name;
}

export type JobSummaryItem = {
  id: string;
  status: string;
  stage: string;
  progress: number;
  created_at: string;
  input: { original_name: string; size_bytes: number };
  processing: { model_name?: string };
  summary?: { scene_count: number } | null;
  error?: string | null;
};

export async function listJobs(): Promise<JobSummaryItem[]> {
  const response = await fetch(`${API_BASE}/api/jobs/`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.jobs as JobSummaryItem[];
}

export async function createCompareFromUpload(
  file: File,
  presetA: string,
  presetB: string,
): Promise<{ job_a: RefineShotJob; job_b: RefineShotJob }> {
  const form = new FormData();
  form.append("file", file);
  form.append("preset_a", presetA);
  form.append("preset_b", presetB);

  const response = await fetch(`${API_BASE}/api/compare/from-upload`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
  return response.json();
}

export async function listModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.models as ModelInfo[];
}

export async function createJobFromUpload(file: File, preset: string): Promise<RefineShotJob> {
  const form = new FormData();
  form.append("file", file);
  form.append("preset", preset);

  const response = await fetch(`${API_BASE}/api/jobs/from-upload`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }

  return response.json();
}

export async function getJob(jobId: string): Promise<RefineShotJob> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
  return response.json();
}

export async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
}

export function exportUrl(jobId: string, kind: "json" | "csv" | "txt"): string {
  return `${API_BASE}/api/jobs/${jobId}/exports/${kind}`;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}
