import { useRef, useState, useEffect } from "react";
import { css } from "../../styled-system/css/css.mjs";
import { CAPTION_FONT, FOCUSED_DESC } from "../constants";

export function MiddleArrowColumn({
  width = 300,
  label = FOCUSED_DESC,
}: {
  width?: number;
  label?: string;
}) {
  const colRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: width, h: 0 });

  useEffect(() => {
    const el = colRef.current;
    if (!el) return;

    const update = () => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    };
    update();

    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    ro?.observe(el);
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("resize", update);
      ro?.disconnect();
    };
  }, []);

  const { w, h } = size;
  const margin = 20;
  const startX = margin;
  const endX = Math.max(margin, w - margin);
  const midY = Math.max(0, h * 0.55);
  const yUp = Math.max(0, midY - Math.min(0.35 * h, 180));
  const yDown = Math.min(h, midY + Math.min(0.28 * h, 150));

  // arrowhead geometry
  const markerW = 16;
  const markerH = 12;

  const endBaseX = endX - markerW + 0.5;

  const d =
    `M ${startX} ${midY}` +
    ` C ${w * 0.30} ${midY}, ${w * 0.38} ${yUp}, ${w * 0.48} ${midY - 6}` +
    ` S ${w * 0.62} ${yDown}, ${w * 0.73} ${midY + 6}` +
    ` S ${endBaseX - 10} ${midY - 2}, ${endBaseX} ${midY}`;

  return (
    <div
      ref={colRef}
      className={css({
        width,
        minWidth: width,
        maxWidth: width,
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      })}
      style={{
        background: "#fff",
        borderLeft: "1px solid #E5E7EB",
        borderRight: "1px solid #E5E7EB",
      }}
    >
      {/* Arrow */}
      <svg
        width="100%"
        height="100%"
        style={{ position: "absolute", inset: 0, overflow: "visible", pointerEvents: "none" }}
      >
        <defs>
          {/* Arrowhead whose BASE is at x=0; attach that base to the path end */}
          <marker
            id="arrowhead"
            viewBox={`0 0 ${markerW} ${markerH}`}
            markerWidth={markerW}
            markerHeight={markerH}
            markerUnits="userSpaceOnUse"
            refX="0"
            refY={(markerH / 2).toString()}
            orient="auto"
          >
            <path d={`M0,0 L${markerW},${markerH / 2} L0,${markerH} Z`} fill="#9CA3AF" />
          </marker>
        </defs>
        <path
          d={d}
          stroke="#9CA3AF"
          strokeWidth={2}
          fill="none"
          strokeLinecap="round"
          markerEnd="url(#arrowhead)"
          style={{ filter: "drop-shadow(0 0 2px rgba(0,0,0,0.10))" }}
        />
      </svg>

      {/* Label bubble near the top center */}
      <div
        style={{
          position: "absolute",
          top: 12,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(255,255,255,0.96)",
          border: "1px solid #E5E7EB",
          borderRadius: 8,
          padding: "6px 10px",
          fontSize: 12,
          color: "#374151",
          lineHeight: 1.2,
          pointerEvents: "none",
          boxShadow: "0 1px 2px rgba(0,0,0,0.06)",
          textAlign: "center",
          maxWidth: Math.max(240, width - 40),
          whiteSpace: "pre-wrap",
          fontFamily: CAPTION_FONT,
        }}
      >
        {label}
      </div>
    </div>
  );
}
