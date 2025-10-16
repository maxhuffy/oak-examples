import { useState, useMemo } from "react";
import { Button } from "@luxonis/common-fe-components";
import { css } from "../styled-system/css/css.mjs";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { useNotifications } from "./Notifications.tsx";

export function SnapCollectionButton() {
  const connection = useConnection();
  const { notify } = useNotifications();

  // master run/stop
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);

  // timing toggle + interval (only visible when toggle is ON)
  const [timingEnabled, setTimingEnabled] = useState(false);
  const [timeIntervalStr, setTimeIntervalStr] = useState("60");
  const timeInterval = useMemo(() => Number.parseInt(timeIntervalStr, 10), [timeIntervalStr]);
  const timeIntervalIsValid = Number.isInteger(timeInterval) && timeInterval > 0;

  // no-detections toggle
  const [noDetEnabled, setNoDetEnabled] = useState(false);
  // new-detections (tracker) toggle
  const [newDetEnabled, setNewDetEnabled] = useState(false);

  const postConfig = (runFlag: boolean) => {
    const payload = runFlag
      ? {
          timed: { enabled: timingEnabled, interval: timingEnabled ? timeInterval : 0 },
          noDetections: { enabled: noDetEnabled },
          newDetections: { enabled: newDetEnabled },
        }
      : {
          timed: { enabled: false, interval: 0 },
          noDetections: { enabled: false },
          newDetections: { enabled: false },
        };

    (connection as any).daiConnection?.postToService(
      "Snap Collection Service",
      payload,
      (_resp: any) => {
        setBusy(false);
        setRunning(runFlag);
        notify(runFlag ? "Snapping started." : "Snapping stopped.", { type: "success" });
      }
    );
  };

  const handleStartStop = () => {
    if (!connection.connected) {
      notify("Not connected to device.", { type: "error" });
      return;
    }
    if (busy) return;

    if (!running && timingEnabled && !timeIntervalIsValid) {
      notify("Please enter a positive integer time interval (seconds).", { type: "error" });
      return;
    }

    setBusy(true);
    if (!running) {
      notify("Starting snapping…", { type: "info" });
      postConfig(true);
    } else {
      notify("Stopping snapping…", { type: "info" });
      postConfig(false);
    }
  };

  const disabledControls = busy || running;

  return (
    <div className={css({ display: "flex", flexDirection: "column", gap: "sm" })}>
      <div className={css({ width: "full", height: "2px", backgroundColor: "gray.300", my: "sm" })} />

      {/* Timing switch (checkbox) */}
      <div className={css({ display: "flex", alignItems: "center", justifyContent: "space-between" })}>
        <label htmlFor="timingToggle" className={css({ fontWeight: "semibold" })}>
          Timing
        </label>
        <input
          id="timingToggle"
          type="checkbox"
          checked={timingEnabled}
          onChange={(e) => setTimingEnabled(e.target.checked)}
          disabled={disabledControls}
          className={css({
            width: "5",
            height: "5",
            cursor: disabledControls ? "not-allowed" : "pointer",
          })}
        />
      </div>

      {/* Interval input — only when timing is ON */}
      {timingEnabled && (
        <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
          <span className={css({ fontWeight: "medium" })}>Minimum interval (seconds)</span>
          <div className={css({ display: "flex", alignItems: "center", gap: "sm" })}>
            <input
              type="number"
              min={1}
              step={1}
              inputMode="numeric"
              pattern="\\d*"
              value={timeIntervalStr}
              onChange={(e) => setTimeIntervalStr(e.target.value)}
              disabled={disabledControls}
              className={css({
                flex: "1",
                px: "sm",
                py: "xs",
                borderWidth: "1px",
                borderColor: disabledControls
                  ? "gray.300"
                  : timeIntervalIsValid
                  ? "gray.300"
                  : "red.500",
                rounded: "md",
                _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
              })}
              aria-invalid={!timeIntervalIsValid && !disabledControls}
              aria-label="Time interval in seconds"
            />
            <span className={css({ color: "gray.600" })}>seconds</span>
          </div>
        </label>
      )}

      {/* No detections switch */}
      <div className={css({ display: "flex", alignItems: "center", justifyContent: "space-between" })}>
        <label htmlFor="noDetToggle" className={css({ fontWeight: "semibold" })}>
          No detections
        </label>
        <input
          id="noDetToggle"
          type="checkbox"
          checked={noDetEnabled}
          onChange={(e) => setNoDetEnabled(e.target.checked)}
          disabled={disabledControls}
          className={css({
            width: "5",
            height: "5",
            cursor: disabledControls ? "not-allowed" : "pointer",
          })}
        />
      </div>

      {/* New detections switch */}
      <div className={css({ display: "flex", alignItems: "center", justifyContent: "space-between" })}>
        <label htmlFor="newDetToggle" className={css({ fontWeight: "semibold" })}>
          New detections
        </label>
        <input
          id="newDetToggle"
          type="checkbox"
          checked={newDetEnabled}
          onChange={(e) => setNewDetEnabled(e.target.checked)}
          disabled={disabledControls}
          className={css({
            width: "5",
            height: "5",
            cursor: disabledControls ? "not-allowed" : "pointer",
          })}
        />
      </div>

      {/* Start / Stop */}
      <Button
        onClick={handleStartStop}
        disabled={busy || (!running && timingEnabled && !timeIntervalIsValid)}
        className={css({
          width: "full",
          py: "sm",
          fontWeight: "semibold",
          backgroundColor: running ? "red.600" : "blue.600",
          color: "white",
          _hover: { backgroundColor: running ? "red.700" : "blue.700" },
          _active: { backgroundColor: running ? "red.800" : "blue.800" },
          _disabled: { opacity: 0.6, cursor: "not-allowed" },
        })}
      >
        {busy ? (running ? "Stopping…" : "Starting…") : (running ? "Stop Snapping" : "Start Snapping")}
      </Button>
    </div>
  );
}
