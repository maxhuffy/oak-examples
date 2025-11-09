# Camera Offset Explained

## What is CAMERA_OFFSET_Y_MM?

It's the vertical distance between the **camera lens** and **your eye line** when you're standing at the reference distance looking at the mirror.

## How to Measure It:

1. Stand at your normal viewing distance (e.g., 500mm from the mirror)
2. Look straight ahead at where your eyes would be in the reflection
3. Note where the camera lens is positioned vertically
4. Measure the distance between:
   - Your eye level (horizontal line through your eyes)
   - The camera lens center

## Visual Diagram:

```
Side View (you looking at mirror):

                    🎥 Camera Lens
                     |
                     |← CAMERA_OFFSET_Y_MM (150mm in this example)
                     |
    You → 👤 ━━━━━━━━━━━━━━━━━ ← Your Eye Line (this is the reference point!)
          👁️ Your Eyes
                     |
                     |
          ━━━━━━━━━━━━━━━━━━━━━ ← Mirror surface (behind display)
              Two-way Mirror
          ┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃
          ┃    Display Screen  ┃
          ┃                    ┃
```

## Why This Matters:

### Real Mirror Physics:
- In a real mirror, you see your reflection at YOUR eye level
- Your eyes are the "camera" looking at the mirror

### Magic Mirror Problem:
- The camera is NOT at your eye level (it's typically above)
- Camera sees you from a DIFFERENT angle than your eyes see the mirror
- This creates a parallax mismatch

### The Solution:
- Shift the camera's view UP to compensate for it being above your eyes
- The amount to shift depends on:
  1. How far above your eyes the camera is (CAMERA_OFFSET_Y_MM)
  2. How far you are from the mirror (distance_ratio)
  3. The current scale factor (combined_scale)

## Example Calculation:

**Setup:**
- Camera is 150mm above your eye line
- You're standing at 500mm (reference distance)
- PIXELS_PER_MM = 0.2
- combined_scale = 8.0 (display scaling)

**Math:**
```
distance_ratio = 500 / 500 = 1.0 (at reference distance)
y_offset_pixels = -150 × 0.2 × 1.0 × 8.0 = -240 pixels

The NEGATIVE value means shift UP by 240 pixels
```

**If you move closer (250mm):**
```
distance_ratio = 500 / 250 = 2.0 (parallax effect doubles!)
y_offset_pixels = -150 × 0.2 × 2.0 × 8.0 = -480 pixels

Shift UP by 480 pixels (offset increases when closer)
```

**If you move farther (1000mm):**
```
distance_ratio = 500 / 1000 = 0.5 (parallax effect halves)
y_offset_pixels = -150 × 0.2 × 0.5 × 8.0 = -120 pixels

Shift UP by 120 pixels (offset decreases when farther)
```

## Common Scenarios:

### Camera Above Your Eyes (Typical Setup):
- CAMERA_OFFSET_Y_MM = **positive** (e.g., 150)
- Calculation produces **negative** offset (shift UP)
- ✅ Correct - compensates for camera being too high

### Camera Below Your Eyes (Unusual):
- CAMERA_OFFSET_Y_MM = **negative** (e.g., -100)
- Calculation produces **positive** offset (shift DOWN)
- ✅ Correct - compensates for camera being too low

### Camera At Eye Level (Ideal but Impractical):
- CAMERA_OFFSET_Y_MM = 0
- No offset applied
- Image appears exactly where camera sees it

## Calibration Tips:

1. **Start with a measurement**: Use a ruler/tape measure to find actual camera height above your eyes

2. **Use the calibration mode**: Press 'C' in the app and use arrow keys to fine-tune

3. **Adjust PIXELS_PER_MM**: 
   - If offset effect is too strong → decrease (try 0.15)
   - If offset effect is too weak → increase (try 0.25)

4. **Check at different distances**:
   - Stand at reference distance (500mm) - should align perfectly
   - Move closer - reflection should still track correctly
   - Move farther - reflection should still track correctly
   - If alignment breaks at different distances, adjust PIXELS_PER_MM

## The Key Insight:

**It's NOT about where the camera is on the mirror frame.**  
**It's about where the camera is relative to YOUR EYES.**

When you look in a real mirror, your eyes are the "camera." The magic mirror's camera is in a different position than your eyes, so we need to correct for that offset.
