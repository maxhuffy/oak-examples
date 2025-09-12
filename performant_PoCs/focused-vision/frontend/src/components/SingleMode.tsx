import React from "react";
import { css } from "../../styled-system/css/css.mjs";
import { Streams } from "@luxonis/depthai-viewer-common";
import { topicGroups, FOCUSED_TOPICS, NON_FOCUSED_TOPICS } from "../constants";
import { MiddleArrowColumn } from "./MiddleArrowColumn";
import { NonFocusedSidebar } from "./NonFocusedSidebar";

const PatchedStreams = Streams as unknown as React.ComponentType<any>;

interface SingleModeProps {
  mode: "focused" | "nonFocused";
  connected: boolean;
  onModeToggle: () => void;
}

export function SingleMode({ mode, connected, onModeToggle }: SingleModeProps) {
  const defaultTopics = mode === "focused" ? FOCUSED_TOPICS : NON_FOCUSED_TOPICS;

  return (
    <>
      {mode === "focused" ? (
        // Two Streams with a middle column for padding & arrow.
        <div
          className={css({
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "row",
          })}
        >
          <section className={css({ flex: 1, minWidth: 0 })}>
            <PatchedStreams
              key={`single-left-${connected ? "on" : "off"}`}
              topicGroups={topicGroups}
              defaultTopics={FOCUSED_TOPICS}
              allowedTopics={["Video"]}
            />
          </section>

          <MiddleArrowColumn width={300} />

          <section className={css({ flex: 1, minWidth: 0 })}>
            <PatchedStreams
              key={`single-right-${connected ? "on" : "off"}`}
              topicGroups={topicGroups}
              allowedTopics={["Crops Mosaic"]}
            />
          </section>
        </div>
      ) : (
        <div
          className={css({
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "row",
            gap: "md",
          })}
        >
          <div className={css({ flex: 1, minHeight: 0 })}>
            <Streams
              key={`single-${mode}-${connected ? "on" : "off"}`}
              topicGroups={topicGroups}
              defaultTopics={defaultTopics}
            />
          </div>

          <NonFocusedSidebar />
        </div>
      )}

      <footer>
        <button
          onClick={onModeToggle}
          className={css({
            width: "100%",
            paddingY: "md",
            borderRadius: "lg",
            borderWidth: "1px",
            borderColor: "gray.300",
            backgroundColor: "transparent",
            _hover: { backgroundColor: "gray.100" },
            fontWeight: "600",
          })}
        >
          {mode === "focused" ? "Switch to Non-Focused" : "Switch to Focused"}
        </button>
      </footer>
    </>
  );
}
