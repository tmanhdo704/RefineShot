import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

export type Backend = "auto" | "phase2" | "baseline";

type Props = {
  backend: Backend;
  disabled?: boolean;
  onChange: (backend: Backend) => void;
};

export default function AdvancedSettings({ backend, disabled, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="advanced-settings">
      <button type="button" className="advanced-toggle" onClick={() => setOpen((value) => !value)} disabled={disabled}>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        Advanced settings
      </button>

      {open && (
        <div className="advanced-body">
          <label className="advanced-field">
            <span>Backend</span>
            <select
              value={backend}
              disabled={disabled}
              onChange={(event) => onChange(event.target.value as Backend)}
            >
              <option value="auto">Auto (server default)</option>
              <option value="phase2">RefineShot</option>
              <option value="baseline">OpenCV baseline</option>
            </select>
          </label>
        </div>
      )}
    </div>
  );
}
