import React from "react";
import { css } from "../../styled-system/css/css.mjs";
import { Streams } from "@luxonis/depthai-viewer-common";
import { topicGroups, defaultOpenAll, sortAB, FOCUSED_DESC, NON_FOCUSED_DESC, CAPTION_FONT } from "../constants";

const PatchedStreams = Streams as unknown as React.ComponentType<any>;

interface SplitModeProps {
  connected: boolean;
}

export function SplitMode({ connected }: SplitModeProps) {
  return (
    <div
      className={css({
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "row",
        gap: "md",
      })}
    >
      {/* LEFT: Focused (A + B) */}
      <section
        className={css({
          flex: 2,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          gap: "sm",
        })}
      >
        <h2 className={css({ fontSize: "lg", fontWeight: "700" })}>Focused</h2>

        {/* caption for focused */}
        <p
          className={css({ fontSize: "sm", color: "gray.600", marginBottom: "2" })}
          style={{ fontFamily: CAPTION_FONT }}
        >
          {FOCUSED_DESC}
        </p>

        <div className={css({ flex: 1, minHeight: 0 })}>
          <PatchedStreams
            key={`left-${connected ? "on" : "off"}`}
            topicGroups={topicGroups}
            defaultTopics={defaultOpenAll}
            allowedTopics={["Video", "Crops Mosaic"]}
            topicSortingFunction={sortAB}
          />
        </div>
      </section>

      {/* Divider */}
      <div className={css({ width: "2px", backgroundColor: "gray.300" })} />

      {/* RIGHT: Non Focused (C) */}
      <section
        className={css({
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          gap: "sm",
        })}
      >
        <h2 className={css({ fontSize: "lg", fontWeight: "700" })}>Non Focused</h2>

        {/* caption for non-focused */}
        <p
          className={css({ fontSize: "sm", color: "gray.600", marginBottom: "2" })}
          style={{ fontFamily: CAPTION_FONT }}
        >
          {NON_FOCUSED_DESC}
        </p>

        <div className={css({ flex: 1, minHeight: 0 })}>
          <PatchedStreams
            key={`right-${connected ? "on" : "off"}`}
            topicGroups={topicGroups}
            allowedTopics={["Non Focused Video"]}
          />
        </div>
      </section>
    </div>
  );
}
