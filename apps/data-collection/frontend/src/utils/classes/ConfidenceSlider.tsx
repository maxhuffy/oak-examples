import { useState, useEffect } from "react";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { SliderControl } from "../SliderControl.tsx";

interface ConfidenceSliderProps {
  initialValue?: number;
  disabled?: boolean;
}

export function ConfidenceSlider({ initialValue = 0.5, disabled }: ConfidenceSliderProps) {
  const connection = useConnection();
  const [value, setValue] = useState(initialValue);

  // Update value from backend config
  useEffect(() => {
    if (initialValue !== undefined && Number.isFinite(initialValue)) {
      console.log("[ConfidenceSlider] Restoring value from backend:", initialValue);
      setValue(initialValue);
    }
  }, [initialValue]);

  const handleCommit = (v: number) => {
    if (Number.isFinite(v)) {
      connection.daiConnection?.postToService(
        // @ts-ignore - Custom service
        "Threshold Update Service",
        v,
        (resp: any) => console.log("[ConfidenceSlider] BE ack:", resp)
      );
    }
  };

  return (
    <SliderControl
      label={`Confidence Threshold: ${(value * 100).toFixed(0)}%`}
      value={value}
      onChange={setValue}
      onCommit={handleCommit}
      min={0}
      max={1}
      step={0.01}
      disabled={disabled}
      aria-label="Confidence threshold"
    />
  );
}
