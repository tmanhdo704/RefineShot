import { Download, FileJson, FileText, Table } from "lucide-react";

import { RefineShotJob, mediaUrl } from "../api/client";
import { exportUrl } from "../api/jobs";

type Props = {
  job: RefineShotJob;
};

export default function ExportPanel({ job }: Props) {
  const links = [
    { label: "JSON", icon: FileJson, href: job.exports?.json_url ? mediaUrl(job.exports.json_url) : exportUrl(job.id, "json") },
    { label: "CSV", icon: Table, href: job.exports?.csv_url ? mediaUrl(job.exports.csv_url) : exportUrl(job.id, "csv") },
    { label: "TXT", icon: FileText, href: job.exports?.txt_url ? mediaUrl(job.exports.txt_url) : exportUrl(job.id, "txt") }
  ];

  return (
    <section className="panel exports-panel">
      <div className="panel-heading">
        <span>Export</span>
        <Download size={18} />
      </div>
      <div className="export-links">
        {links.map((link) => {
          const Icon = link.icon;
          return (
            <a key={link.label} href={link.href} target="_blank" rel="noreferrer">
              <Icon size={16} />
              {link.label}
            </a>
          );
        })}
      </div>
    </section>
  );
}
