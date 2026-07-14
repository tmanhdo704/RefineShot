import { Film, UploadCloud } from "lucide-react";
import { DragEvent, useRef, useState } from "react";

import { formatBytes } from "../api/client";

type Props = {
  file: File | null;
  disabled?: boolean;
  onFileChange: (file: File | null) => void;
};

export default function UploadDropzone({ file, disabled, onFileChange }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const nextFile = event.dataTransfer.files?.[0] || null;
    if (nextFile) onFileChange(nextFile);
  };

  return (
    <div
      className={`dropzone ${isDragging ? "is-dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/*"
        disabled={disabled}
        onChange={(event) => onFileChange(event.target.files?.[0] || null)}
      />
      <div className="dropzone-icon">{file ? <Film size={28} /> : <UploadCloud size={30} />}</div>
      <div className="dropzone-copy">
        <strong>{file ? file.name : "Drop a video here"}</strong>
        <span>{file ? formatBytes(file.size) : "MP4, MOV, WEBM, MKV"}</span>
      </div>
    </div>
  );
}
