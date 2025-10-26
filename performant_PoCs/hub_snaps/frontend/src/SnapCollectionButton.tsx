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

  // --- helpers for validation/state ---
  const hasTooManyDecimals = (s: string) => /\.\d{2,}$/.test(s.trim());
  const oneDecimalOrInt = (s: string) => /^\d+(\.\d{0,1})?$/.test(s.trim());
  const asFloat = (s: string) => Number.parseFloat(s.trim());

  // timing (minutes, allow one decimal; default 5.0)
  const [timingEnabled, setTimingEnabled] = useState(false);
  const [timeIntervalStr, setTimeIntervalStr] = useState("5.0");
  const timeIntervalMin = useMemo(() => asFloat(timeIntervalStr), [timeIntervalStr]);
  const timeIntervalValid =
    (!timingEnabled) ||
    (timeIntervalStr !== "" && oneDecimalOrInt(timeIntervalStr) && Number.isFinite(timeIntervalMin) && timeIntervalMin > 0);

  // no detections (minutes, allow one decimal)
  const [noDetEnabled, setNoDetEnabled] = useState(false);
  const [noDetCooldownStr, setNoDetCooldownStr] = useState("15.0");
  const noDetCooldownMin = useMemo(() => asFloat(noDetCooldownStr), [noDetCooldownStr]);
  const noDetCooldownValid =
    (!noDetEnabled) ||
    (noDetCooldownStr !== "" && oneDecimalOrInt(noDetCooldownStr) && Number.isFinite(noDetCooldownMin) && noDetCooldownMin >= 0);

  // low confidence (cooldown minutes, allow one decimal; threshold unchanged)
  const [lowConfEnabled, setLowConfEnabled] = useState(false);
  const [lowConfCooldownStr, setLowConfCooldownStr] = useState("15.0");
  const lowConfCooldownMin = useMemo(() => asFloat(lowConfCooldownStr), [lowConfCooldownStr]);
  const lowConfCooldownValid =
    (!lowConfEnabled) ||
    (lowConfCooldownStr !== "" && oneDecimalOrInt(lowConfCooldownStr) && Number.isFinite(lowConfCooldownMin) && lowConfCooldownMin >= 0);
  const [lowConfThreshold, setLowConfThreshold] = useState(0.30); // slider 0..1

  // lost in middle (cooldown minutes, allow one decimal; margin unchanged)
  const [lostMidEnabled, setLostMidEnabled] = useState(false);
  const [lostMidCooldownStr, setLostMidCooldownStr] = useState("15.0");
  const lostMidCooldownMin = useMemo(() => asFloat(lostMidCooldownStr), [lostMidCooldownStr]);
  const lostMidCooldownValid =
    (!lostMidEnabled) ||
    (lostMidCooldownStr !== "" && oneDecimalOrInt(lostMidCooldownStr) && Number.isFinite(lostMidCooldownMin) && lostMidCooldownMin >= 0);
  const [lostMidMarginStr, setLostMidMarginStr] = useState("0.20"); // fraction 0..0.49
  const lostMidMargin = useMemo(() => Number.parseFloat(lostMidMarginStr), [lostMidMarginStr]);
  const lostMidMarginValid = !Number.isNaN(lostMidMargin) && lostMidMargin >= 0 && lostMidMargin <= 0.49;

  const anyInvalid =
    !timeIntervalValid || !noDetCooldownValid || !lowConfCooldownValid || !lostMidCooldownValid || !lostMidMarginValid;

  const Divider = () => (
    <div className={css({ width: "full", height: "1px", backgroundColor: "gray.200", my: "sm" })} />
  );

  // helper: post to BE
  const postToService = (payload: any, onDone?: () => void) => {
    (connection as any).daiConnection?.postToService("Snap Collection Service", payload, (_resp: any) => {
      onDone?.();
    });
  };

  // compose full payload when starting/stopping (convert minutes -> seconds)
  const postConfig = (runFlag: boolean) => {
    const payload = runFlag
      ? {
          timed: {
            enabled: timingEnabled,
            interval: timingEnabled ? timeIntervalMin * 60 : 0,
          },
          noDetections: {
            enabled: noDetEnabled,
            cooldown: noDetEnabled ? noDetCooldownMin * 60 : undefined,
          },
          lowConfidence: lowConfEnabled
            ? {
                enabled: true,
                threshold: lowConfThreshold,
                cooldown: lowConfCooldownMin * 60,
              }
            : { enabled: false },
          lostMid: lostMidEnabled
            ? {
                enabled: true,
                cooldown: lostMidCooldownMin * 60,
                margin: lostMidMargin,
              }
            : { enabled: false },
        }
      : {
          timed: { enabled: false, interval: 0 },
          noDetections: { enabled: false },
          lowConfidence: { enabled: false },
          lostMid: { enabled: false },
        };

    postToService(payload, () => {
      setBusy(false);
      setRunning(runFlag);
      notify(runFlag ? "Snapping started." : "Snapping stopped.", { type: "success" });
    });
  };

  // live-update lowConfidence while running (minutes -> seconds)
  const pushLowConfUpdate = () => {
    if (!connection.connected || !running || !lowConfEnabled) return;
    if (!lowConfCooldownValid) return;
    postToService({
      lowConfidence: {
        enabled: true,
        threshold: lowConfThreshold,
        cooldown: lowConfCooldownMin * 60,
      },
    });
  };

  // live-update lostMid while running (minutes -> seconds)
  const pushLostMidUpdate = () => {
    if (!connection.connected || !running || !lostMidEnabled) return;
    if (!lostMidMarginValid || !lostMidCooldownValid) return;
    postToService({
      lostMid: {
        enabled: true,
        cooldown: lostMidCooldownMin * 60,
        margin: lostMidMargin,
      },
    });
  };

  // warn on blur if too many decimals
  const warnIfTooManyDecimals = (label: string, value: string) => {
    if (value.trim() !== "" && hasTooManyDecimals(value)) {
      notify(`${label} allows at most one decimal place.`, { type: "warning" });
    }
  };

  const handleStartStop = () => {
    if (!connection.connected) {
      notify("Not connected to device.", { type: "error" });
      return;
    }
    if (busy) return;

    if (!running) {
      if (!timeIntervalValid && timingEnabled) {
        notify("Please enter a positive timing cooldown (minutes, max 1 decimal).", { type: "error" });
        return;
      }
      if (!noDetCooldownValid && noDetEnabled) {
        notify("Please enter a non-negative no-detections cooldown (minutes, max 1 decimal).", { type: "error" });
        return;
      }
      if (lowConfEnabled) {
        if (!lowConfCooldownValid) {
          notify("Please enter a non-negative low-confidence cooldown (minutes, max 1 decimal).", { type: "error" });
          return;
        }
        if (!(lowConfThreshold >= 0 && lowConfThreshold <= 1)) {
          notify("Confidence threshold must be between 0.00 and 1.00.", { type: "error" });
          return;
        }
      }
      if (lostMidEnabled) {
        if (!lostMidCooldownValid) {
          notify("Please enter a non-negative lost-in-middle cooldown (minutes, max 1 decimal).", { type: "error" });
          return;
        }
        if (!lostMidMarginValid) {
          notify("Margin must be a number between 0.00 and 0.49.", { type: "error" });
          return;
        }
      }
    }

    setBusy(true);
    notify(!running ? "Starting snapping…" : "Stopping snapping…", { type: "info" });
    postConfig(!running);
  };

  const disabledControls = busy || running;

  return (
    <div className={css({ display: "flex", flexDirection: "column", gap: "sm" })}>
      <Divider />

      {/* Timing */}
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
          className={css({ width: "5", height: "5", cursor: disabledControls ? "not-allowed" : "pointer" })}
        />
      </div>

      {timingEnabled && (
        <>
          <p className={css({ fontSize: "sm", color: "gray.600" })}>
            Sends a snap periodically; throttled by the cooldown.
          </p>
          <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
            <span className={css({ fontWeight: "medium" })}>Cooldown (minutes)</span>
            <div className={css({ display: "flex", alignItems: "center", gap: "sm" })}>
              <input
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                value={timeIntervalStr}
                onChange={(e) => setTimeIntervalStr(e.target.value)}
                onBlur={() => warnIfTooManyDecimals("Timing cooldown", timeIntervalStr)}
                disabled={disabledControls}
                className={css({
                  flex: "1",
                  px: "sm",
                  py: "xs",
                  borderWidth: "1px",
                  borderColor:
                    disabledControls
                      ? "gray.300"
                      : timeIntervalValid
                      ? "gray.300"
                      : "red.500",
                  rounded: "md",
                  _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
                })}
                aria-invalid={!timeIntervalValid && !disabledControls}
                aria-label="Timing cooldown (minutes, max 1 decimal)"
              />
              <span className={css({ color: "gray.600" })}>minutes</span>
            </div>
            {!timeIntervalValid && (
              <span className={css({ fontSize: "xs", color: "red.600" })}>
                Enter a positive number with at most one decimal place.
              </span>
            )}
          </label>
        </>
      )}

      <Divider />

      {/* No detections */}
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
          className={css({ width: "5", height: "5", cursor: disabledControls ? "not-allowed" : "pointer" })}
        />
      </div>

      {noDetEnabled && (
        <>
          <p className={css({ fontSize: "sm", color: "gray.600" })}>
            Sends a snap when a frame has zero detections; throttled by the cooldown.
          </p>
          <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
            <span className={css({ fontWeight: "medium" })}>Cooldown (minutes)</span>
            <div className={css({ display: "flex", alignItems: "center", gap: "sm" })}>
              <input
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                value={noDetCooldownStr}
                onChange={(e) => setNoDetCooldownStr(e.target.value)}
                onBlur={() => warnIfTooManyDecimals("No-detections cooldown", noDetCooldownStr)}
                disabled={disabledControls}
                className={css({
                  flex: "1",
                  px: "sm",
                  py: "xs",
                  borderWidth: "1px",
                  borderColor:
                    disabledControls
                      ? "gray.300"
                      : noDetCooldownValid
                      ? "gray.300"
                      : "red.500",
                  rounded: "md",
                  _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
                })}
                aria-invalid={!noDetCooldownValid && !disabledControls}
                aria-label="No detections cooldown (minutes, max 1 decimal)"
              />
              <span className={css({ color: "gray.600" })}>minutes</span>
            </div>
            {!noDetCooldownValid && (
              <span className={css({ fontSize: "xs", color: "red.600" })}>
                Enter a non-negative number with at most one decimal place.
              </span>
            )}
          </label>
        </>
      )}

      <Divider />

      {/* Low confidence */}
      <div className={css({ display: "flex", alignItems: "center", justifyContent: "space-between" })}>
        <label htmlFor="lowConfToggle" className={css({ fontWeight: "semibold" })}>
          Low confidence
        </label>
        <input
          id="lowConfToggle"
          type="checkbox"
          checked={lowConfEnabled}
          onChange={(e) => setLowConfEnabled(e.target.checked)}
          disabled={disabledControls}
          className={css({ width: "5", height: "5", cursor: disabledControls ? "not-allowed" : "pointer" })}
        />
      </div>

      {lowConfEnabled && (
        <div className={css({ display: "flex", flexDirection: "column", gap: "sm" })}>
          <p className={css({ fontSize: "sm", color: "gray.600" })}>
            Sends a snap if any detection confidence falls below the threshold; throttled by the cooldown.
          </p>

          <label className={css({ fontWeight: "medium" })}>
            Confidence threshold: {(lowConfThreshold * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={lowConfThreshold}
            onChange={(e) => setLowConfThreshold(parseFloat(e.target.value))}
            onMouseUp={pushLowConfUpdate}
            onTouchEnd={pushLowConfUpdate}
            disabled={disabledControls}
            className={css({
              width: "100%",
              appearance: "none",
              height: "4px",
              borderRadius: "full",
              backgroundColor: "gray.300",
              "&::-webkit-slider-thumb": {
                appearance: "none",
                width: "12px",
                height: "12px",
                borderRadius: "full",
                backgroundColor: "blue.500",
                cursor: "pointer",
              },
              "&::-moz-range-thumb": {
                appearance: "none",
                width: "12px",
                height: "12px",
                borderRadius: "full",
                backgroundColor: "blue.500",
                cursor: "pointer",
              },
            })}
          />

          <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
            <span className={css({ fontWeight: "medium" })}>Cooldown (minutes)</span>
            <div className={css({ display: "flex", alignItems: "center", gap: "sm" })}>
              <input
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                value={lowConfCooldownStr}
                onChange={(e) => setLowConfCooldownStr(e.target.value)}
                onBlur={() => warnIfTooManyDecimals("Low-confidence cooldown", lowConfCooldownStr)}
                disabled={disabledControls}
                className={css({
                  flex: "1",
                  px: "sm",
                  py: "xs",
                  borderWidth: "1px",
                  borderColor:
                    disabledControls
                      ? "gray.300"
                      : lowConfCooldownValid
                      ? "gray.300"
                      : "red.500",
                  rounded: "md",
                  _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
                })}
                aria-invalid={!lowConfCooldownValid && !disabledControls}
                aria-label="Low confidence cooldown (minutes, max 1 decimal)"
              />
              <span className={css({ color: "gray.600" })}>minutes</span>
            </div>
            {!lowConfCooldownValid && (
              <span className={css({ fontSize: "xs", color: "red.600" })}>
                Enter a non-negative number with at most one decimal place.
              </span>
            )}
          </label>
        </div>
      )}

      <Divider />

      {/* Lost in middle */}
      <div className={css({ display: "flex", alignItems: "center", justifyContent: "space-between" })}>
        <label htmlFor="lostMidToggle" className={css({ fontWeight: "semibold" })}>
          Lost in middle
        </label>
        <input
          id="lostMidToggle"
          type="checkbox"
          checked={lostMidEnabled}
          onChange={(e) => setLostMidEnabled(e.target.checked)}
          disabled={disabledControls}
          className={css({ width: "5", height: "5", cursor: disabledControls ? "not-allowed" : "pointer" })}
        />
      </div>

      {lostMidEnabled && (
        <div className={css({ display: "flex", flexDirection: "column", gap: "sm" })}>
          <p className={css({ fontSize: "sm", color: "gray.600" })}>
            Sends a snap the moment a tracked object disappears inside the center area; throttled by the cooldown.
          </p>

          <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
            <span className={css({ fontWeight: "medium" })}>Edge buffer (each side) — 0.00–0.49</span>
            <input
              type="number"
              min={0}
              max={0.49}
              step={0.01}
              inputMode="decimal"
              value={lostMidMarginStr}
              onChange={(e) => setLostMidMarginStr(e.target.value)}
              onBlur={pushLostMidUpdate}
              disabled={disabledControls}
              className={css({
                px: "sm",
                py: "xs",
                borderWidth: "1px",
                borderColor:
                  disabledControls
                    ? "gray.300"
                    : lostMidMarginValid
                    ? "gray.300"
                    : "red.500",
                rounded: "md",
                _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
              })}
              aria-invalid={!lostMidMarginValid && !disabledControls}
              aria-label="Lost-in-middle margin (0.00–0.49)"
            />
            <span className={css({ fontSize: "xs", color: "gray.600" })}>
              We ignore the outer margin on every edge; only losses inside the remaining center fire snaps.
            </span>
          </label>

          <label className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
            <span className={css({ fontWeight: "medium" })}>Cooldown (minutes)</span>
            <div className={css({ display: "flex", alignItems: "center", gap: "sm" })}>
              <input
                type="number"
                min={0}
                step={0.1}
                inputMode="decimal"
                value={lostMidCooldownStr}
                onChange={(e) => setLostMidCooldownStr(e.target.value)}
                onBlur={() => {
                  warnIfTooManyDecimals("Lost-in-middle cooldown", lostMidCooldownStr);
                  pushLostMidUpdate();
                }}
                disabled={disabledControls}
                className={css({
                  flex: "1",
                  px: "sm",
                  py: "xs",
                  borderWidth: "1px",
                  borderColor:
                    disabledControls
                      ? "gray.300"
                      : lostMidCooldownValid
                      ? "gray.300"
                      : "red.500",
                  rounded: "md",
                  _disabled: { bg: "gray.100", color: "gray.500", cursor: "not-allowed" },
                })}
                aria-invalid={!lostMidCooldownValid && !disabledControls}
                aria-label="Lost-in-middle cooldown (minutes, max 1 decimal)"
              />
              <span className={css({ color: "gray.600" })}>minutes</span>
            </div>
            {!lostMidCooldownValid && (
              <span className={css({ fontSize: "xs", color: "red.600" })}>
                Enter a non-negative number with at most one decimal place.
              </span>
            )}
          </label>
        </div>
      )}

      <Divider />

      <Button
        onClick={handleStartStop}
        disabled={
          busy ||
          anyInvalid ||
          (!running && lowConfEnabled && !(lowConfThreshold >= 0 && lowConfThreshold <= 1))
        }
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
        {busy ? (running ? "Stopping…" : "Starting…") : running ? "Stop Snapping" : "Start Snapping"}
      </Button>
    </div>
  );
}
