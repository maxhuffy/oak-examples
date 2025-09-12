// Use a handwriting-like font only for captions (add the Google Fonts link in index.html).
export const CAPTION_FONT = "'Patrick Hand', 'Caveat', 'Gloria Hallelujah', cursive";

export const topicGroups = {
  Video: "A",
  "Detections Stage 1": "A",
  "Crops Mosaic": "B",
  "Detections Stage 2 Crops": "B",
  "Non Focused Video": "C",
  "Detections Non Focused": "C",
};

// presets
export const FOCUSED_TOPICS = ["Crops Mosaic", "Video"];
export const NON_FOCUSED_TOPICS = ["Non Focused Video"];
export const defaultOpenAll = ["Video", "Crops Mosaic", "Non Focused Video"];

// order helper (used only in split-mode left column)
export const ORDER_AB = ["Video", "Crops Mosaic"];
export const sortAB = (a: { name: string }, b: { name: string }) => {
  const ai = ORDER_AB.indexOf(a.name);
  const bi = ORDER_AB.indexOf(b.name);
  return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
};

// descriptions
export const FOCUSED_DESC =
  "To detect small objects on the image, Focused mode detects larger objects on the full frame downscaled to NN input requirements, creates high-resolution crops from the detections on the original image, and then detects smaller objects within those crops.";
export const NON_FOCUSED_DESC =
  "To detect small objects on the image, Non-Focused mode runs the detection directly on the full frame downscaled to NN input requirements — lighter and simpler, but innacurate.";
