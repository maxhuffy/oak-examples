import { useState, useMemo } from "react";
import { Button } from "@luxonis/common-fe-components";
import { css } from "../styled-system/css/css.mjs";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { useNotifications } from "./Notifications.tsx";

export function SnapCollectionButton() {
  const connection = useConnection();
  const { notify } = useNotifications();

  const [isCollecting, setIsCollecting] = useState(false);
  const [busy, setBusy] = useState(false);

  // Keep as string for forgiving typing; validate on toggle.
  const [timeIntervalStr, setTimeIntervalStr] = useState("60"); // default 60s
  const timeInterval = useMemo(
    () => Number.parseInt(timeIntervalStr, 10),
    [timeIntervalStr]
  );
  const timeIntervalIsValid =
    Number.isInteger(timeInterval) && timeInterval > 0;

  const handleToggle = () => {
    if (!connection.connected) {
      notify("Not connected to device. Unable to toggle collection.", {
        type: "error",
      });
      return;
    }
    if (busy) return;

    const nextState = !isCollecting; // true = start, false = end

    // When starting, require a valid positive integer interval.
    if (nextState && !timeIntervalIsValid) {
      notify("Please enter a positive integer time interval (seconds).", {
        type: "error",
      });
      return;
    }

    setBusy(true);
    notify(
      nextState
        ? `Starting collection (every ${timeInterval}s)…`
        : "Ending collection…",
      { type: "info" }
    );

    // Tuple payload: [startFlag:boolean, interval:number]
    // Start  -> [true, <intervalSeconds>]
    // End    -> [false, -1]
    const payload: [boolean, number] = nextState
      ? [true, timeInterval]
      : [false, -1];

    (connection as any).daiConnection?.postToService(
      "Snap Collection Service",
      payload,
      (resp: any) => {
        console.log("[SnapCollect] Ack:", resp);
        setBusy(false);
        setIsCollecting(nextState);
        notify(
          nextState
            ? `Collection started (every ${timeInterval}s).`
            : "Collection ended.",
          { type: "success" }
        );
      }
    );
  };

  const label = busy
    ? isCollecting
      ? "Ending…"
      : "Starting…"
    : isCollecting
    ? "End Collection"
    : "Start Collection";

  const inputDisabled = busy || isCollecting;

  return (
    <div className={css({ display: "flex", flexDirection: "column", gap: "sm" })}>
      {/* Horizontal divider */}
      <div
        className={css({
          width: "full",
          height: "2px",
          backgroundColor: "gray.300",
          my: "sm",
        })}
      />

      <h1 className={css({ fontSize: "l", fontWeight: "bold" })}>
        Click below to start Snap Collection
      </h1>

      {/* Time interval input (seconds) */}
      <label
        className={css({ display: "flex", flexDirection: "column", gap: "xs" })}
      >
        <span className={css({ fontWeight: "medium" })}>
          What is the minimum time interval between consecutive snaps?
        </span>
        <div
          className={css({
            display: "flex",
            alignItems: "center",
            gap: "sm",
          })}
        >
          <input
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            pattern="\\d*"
            value={timeIntervalStr}
            onChange={(e) => setTimeIntervalStr(e.target.value)}
            disabled={inputDisabled}
            className={css({
              flex: "1",
              px: "sm",
              py: "xs",
              borderWidth: "1px",
              borderColor: inputDisabled
                ? "gray.300"
                : timeIntervalIsValid
                ? "gray.300"
                : "red.500",
              rounded: "md",
              _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
            })}
            aria-invalid={!timeIntervalIsValid && !inputDisabled}
            aria-label="Time interval in seconds"
          />
          <span className={css({ color: "gray.600", whiteSpace: "nowrap" })}>
            seconds
          </span>
        </div>
        <span
          className={css({
            fontSize: "sm",
            color: timeIntervalIsValid ? "gray.600" : "red.600",
          })}
        >
          {timeIntervalIsValid
            ? "Enter a positive integer (in seconds)."
            : "Please enter a positive integer."}
        </span>
      </label>

      {/* Full-width toggle button */}
      <Button
        onClick={handleToggle}
        disabled={busy || (!isCollecting && !timeIntervalIsValid)}
        className={css({
          width: "full",
          py: "sm",
          fontWeight: "semibold",
          backgroundColor: isCollecting ? "red.600" : "blue.600",
          color: "white",
          _hover: { backgroundColor: isCollecting ? "red.700" : "blue.700" },
          _active: { backgroundColor: isCollecting ? "red.800" : "blue.800" },
          _disabled: { opacity: 0.6, cursor: "not-allowed" },
        })}
      >
        {label}
      </Button>
    </div>
  );
}
