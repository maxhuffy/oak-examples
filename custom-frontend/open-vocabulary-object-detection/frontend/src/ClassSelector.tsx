import { css } from "../styled-system/css/css.mjs";
import { useRef, useState } from "react";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { useNotifications } from "./Notifications.tsx";

interface ClassSelectorProps {
  initialClasses?: string[];
  onClassesUpdated?: (classes: string[]) => void;
}

export function ClassSelector({ initialClasses = [], onClassesUpdated }: ClassSelectorProps) {
  const connection = useConnection();
  const { notify } = useNotifications();
  const inputRef = useRef<HTMLInputElement>(null);
  const [classes, setClasses] = useState<string[]>(initialClasses);

  const handleCommit = () => {
    if (!inputRef.current) return;

    const value = inputRef.current.value;
    const updated = value
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);

    if (updated.length === 0) {
      notify("Please enter at least one class (comma separated).", {
        type: "warning",
        durationMs: 5000,
      });
      return;
    }

    if (!connection.connected) {
      notify("Not connected to device. Unable to update classes.", {
        type: "error",
      });
      return;
    }

    console.log("Sending class update to backend:", updated);
    notify(`Updating ${updated.length} class${updated.length > 1 ? "es" : ""}…`, {
      type: "info",
    });

    connection.daiConnection?.postToService(
      // @ts-ignore - Custom service
      "Class Update Service",
      updated,
      () => {
        console.log("Backend acknowledged class update");
        setClasses(updated);
        onClassesUpdated?.(updated);
        notify(`Classes updated (${updated.join(", ")})`, {
          type: "success",
          durationMs: 6000,
        });
      }
    );

    inputRef.current.value = "";
  };

  return (
    <div className={css({ display: "flex", flexDirection: "column", gap: "xs" })}>
      <label className={css({ fontWeight: "medium" })}>Tracked Classes:</label>

      {/* Current classes display */}
      <div
        className={css({
          border: "1px solid token(colors.border.subtle)",
          borderRadius: "md",
          backgroundColor: "token(colors.bg.surface)",
          padding: "sm",
          maxHeight: "120px",
          overflowY: "auto",
        })}
      >
        {classes.length > 0 ? (
          <ul className={css({ listStyle: "disc", pl: "lg", m: 0 })}>
            {classes.map((cls, i) => (
              <li key={i}>{cls}</li>
            ))}
          </ul>
        ) : (
          <p className={css({ color: "gray.600", fontSize: "sm" })}>No classes selected.</p>
        )}
      </div>

      {/* Input field */}
      <input
        ref={inputRef}
        type="text"
        placeholder="person,chair,TV"
        onKeyDown={(e) => e.key === "Enter" && handleCommit()}
        className={css({
          border: "1px solid token(colors.border.subtle)",
          borderRadius: "md",
          paddingX: "sm",
          paddingY: "xs",
          fontSize: "sm",
          backgroundColor: "token(colors.bg.input)",
        })}
      />

      {/* Update button */}
      <button
        onClick={handleCommit}
        className={css({
          mt: "xs",
          paddingY: "xs",
          borderRadius: "md",
          backgroundColor: "blue.500",
          color: "white",
          fontWeight: "medium",
          border: "none",
          cursor: "pointer",
          _hover: { backgroundColor: "blue.600" },
          transition: "background-color 0.15s ease",
        })}
      >
        Update Classes
      </button>
    </div>
  );
}
