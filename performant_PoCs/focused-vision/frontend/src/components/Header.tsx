import { css } from "../../styled-system/css/css.mjs";

type LayoutMode = "split" | "single";

interface HeaderProps {
  layout: LayoutMode;
  mode: "focused" | "nonFocused";
  onLayoutToggle: () => void;
}

export function Header({ layout, mode, onLayoutToggle }: HeaderProps) {
  const headerRight = (
    <button
      onClick={onLayoutToggle}
      className={css({
        paddingX: "md",
        paddingY: "xs",
        borderWidth: "1px",
        borderColor: "gray.300",
        borderRadius: "md",
        backgroundColor: "transparent",
        _hover: { backgroundColor: "gray.100" },
        fontWeight: "600",
      })}
    >
      {layout === "split" ? "Single mode" : "Comparison mode"}
    </button>
  );

  return (
    <header
      className={css({
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      })}
    >
      <h1 className={css({ fontSize: "xl", fontWeight: "600" })}>
        {layout === "split"
          ? "Focused & Non Focused"
          : mode === "focused"
          ? "Focused mode"
          : "Non-focused mode"}
      </h1>

      <div className={css({ display: "flex", alignItems: "center", gap: "md" })}>
        <div className={css({ fontSize: "sm", color: "gray.600" })}>
          {layout === "split"
            ? "Showing: Video & Crops Mosaic (left), Non Focused (right)"
            : mode === "focused"
            ? "Showing: Video → Crops Mosaic"
            : "Showing: Non Focused Video"}
        </div>
        {headerRight}
      </div>
    </header>
  );
}
