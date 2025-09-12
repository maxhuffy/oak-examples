import { css } from "../../styled-system/css/css.mjs";
import { NON_FOCUSED_DESC, CAPTION_FONT } from "../constants";

export function NonFocusedSidebar() {
  return (
    <aside
      className={css({
        width: "320px",
        minWidth: "320px",
        maxWidth: "360px",
        padding: "md",
        color: "gray.700",
        lineHeight: "1.4",
      })}
      style={{ borderLeft: "1px solid #E5E7EB", background: "#fff" }}
    >
      <h3 className={css({ fontSize: "md", fontWeight: "700", marginBottom: "2" })}>
        Non-Focused mode
      </h3>
      <p
        className={css({ fontSize: "sm", color: "gray.700" })}
        style={{ fontFamily: CAPTION_FONT }}
      >
        {NON_FOCUSED_DESC}
      </p>
    </aside>
  );
}
