
import { FormEvent, useEffect, useState } from "react";

import UploadDropzone from "../components/UploadDropzone";
import ErrorState from "../components/ErrorState";
import { createJobFromUpload, listModels, modelDisplayName, ModelInfo } from "../api/jobs";

type Props = {
  onJobCreated: (jobId: string) => void;
};

export default function UploadPage({ onJobCreated }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("best_shot");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listModels().then((list) => {
      if (list.length === 0) return;
      setModels(list);
      const def = list.find((m) => m.is_default);
      if (def) setSelectedPreset(def.preset);
    });
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("Choose a video file first.");
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const job = await createJobFromUpload(file, selectedPreset);
      onJobCreated(job.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="page upload-page">
      <form className="upload-panel" onSubmit={handleSubmit}>
        <div className="section-title">
          <span>Phân tích</span>
          <strong>Upload video</strong>
        </div>

        <UploadDropzone file={file} disabled={isSubmitting} onFileChange={setFile} />

        {models.length > 0 && (
          <div className="settings-row">
            <label>
              <span>Model</span>
              <select
                value={selectedPreset}
                disabled={isSubmitting}
                onChange={(e) => setSelectedPreset(e.target.value)}
              >
                {models.filter((m) => m.available).map((m) => (
                  <option key={m.preset} value={m.preset}>
                    {modelDisplayName(m)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}

        {error && <ErrorState message={error} />}

        <button className="primary-action" type="submit" disabled={isSubmitting || !file}>
          {isSubmitting ? "Đang tải lên..." : "Phân tích"}
        </button>

        <p className="upload-hint">Hỗ trợ MP4, MOV, WEBM, MKV · Xuất JSON, CSV, TXT</p>
      </form>
    </main>
  );
}
