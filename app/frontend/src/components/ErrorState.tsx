import { AlertCircle } from "lucide-react";

type Props = {
  title?: string;
  message: string;
};

export default function ErrorState({ title = "Something went wrong", message }: Props) {
  return (
    <section className="error-state">
      <AlertCircle size={20} />
      <div>
        <strong>{title}</strong>
        <span>{message}</span>
      </div>
    </section>
  );
}
