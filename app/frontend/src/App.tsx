import { useEffect, useState } from "react";
import logo from "./assets/logo.jpg";

import ResultPage from "./pages/ResultPage";
import UploadPage from "./pages/UploadPage";
import HistoryPage from "./pages/HistoryPage";
import ComparePage from "./pages/ComparePage";

type Route =
  | { name: "upload" }
  | { name: "result"; jobId: string }
  | { name: "history" }
  | { name: "compare" };

function readRoute(): Route {
  if (window.location.pathname === "/history") return { name: "history" };
  if (window.location.pathname === "/compare") return { name: "compare" };
  const match = window.location.pathname.match(/^\/result\/([^/]+)/);
  if (match) return { name: "result", jobId: match[1] };
  return { name: "upload" };
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => readRoute());

  useEffect(() => {
    const onPop = () => setRoute(readRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigateToResult = (jobId: string) => {
    window.history.pushState({}, "", `/result/${jobId}`);
    setRoute({ name: "result", jobId });
  };

  const navigateHome = () => {
    window.history.pushState({}, "", "/");
    setRoute({ name: "upload" });
  };

  const navigateHistory = () => {
    window.history.pushState({}, "", "/history");
    setRoute({ name: "history" });
  };

  const navigateCompare = () => {
    window.history.pushState({}, "", "/compare");
    setRoute({ name: "compare" });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-button" type="button" onClick={navigateHome}>
          <img src={logo} alt="Logo" className="brand-logo" />
          <span>
            <strong>KHÓA LUẬN TỐT NGHIỆP</strong>
            <small>Phát hiện chuyển cảnh bằng mô hình học sâu</small>
          </span>
        </button>
        <nav className="topbar-nav">
          <button
            type="button"
            className={`nav-btn ${route.name === "upload" ? "active" : ""}`}
            onClick={navigateHome}
          >
            Phân tích
          </button>
          <button
            type="button"
            className={`nav-btn ${route.name === "compare" ? "active" : ""}`}
            onClick={navigateCompare}
          >
            So sánh
          </button>
          <button
            type="button"
            className={`nav-btn ${route.name === "history" ? "active" : ""}`}
            onClick={navigateHistory}
          >
            Lịch sử
          </button>
        </nav>
      </header>

      {route.name === "upload" && <UploadPage onJobCreated={navigateToResult} />}
      {route.name === "compare" && <ComparePage />}
      {route.name === "history" && <HistoryPage onSelectJob={navigateToResult} />}
      {route.name === "result" && <ResultPage jobId={route.jobId} onBack={navigateHome} />}
    </div>
  );
}
