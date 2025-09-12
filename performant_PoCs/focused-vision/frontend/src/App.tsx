import { useState } from "react";
import { css } from "../styled-system/css/css.mjs";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { Header, SplitMode, SingleMode } from "./components";


export default function App() {
  type LayoutMode = "split" | "single";
  const [layout, setLayout] = useState<LayoutMode>("single"); // default: Single (Focused)
  const [mode, setMode] = useState<"focused" | "nonFocused">("focused");
  const { connected } = useConnection();

  const handleLayoutToggle = () => {
    setLayout((prev) => (prev === "split" ? "single" : "split"));
  };

  const handleModeToggle = () => {
    setMode((prev) => (prev === "focused" ? "nonFocused" : "focused"));
  };

  return (
    <main
      className={css({
        width: "screen",
        height: "screen",
        display: "flex",
        flexDirection: "column",
        gap: "md",
        padding: "md",
      })}
    >
      <Header
        layout={layout}
        mode={mode}
        onLayoutToggle={handleLayoutToggle}
      />

      {layout === "split" ? (
        <SplitMode connected={connected} />
      ) : (
        <SingleMode
          mode={mode}
          connected={connected}
          onModeToggle={handleModeToggle}
        />
      )}
    </main>
  );
}
