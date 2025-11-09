# Magic Mirror Scaling Math - Complete Breakdown

## What Affects Your Scaling

### 1. **Face Detection & Filtering** (CRITICAL)
**Location**: `magic_mirror.py` lines 250-290 & `utils/roi_from_face.py` lines 30-45

**What happens**:
- YuNet detects faces and returns detections with confidence scores
- **NOW FILTERED**: Only detections with confidence ≥ 0.75 are used
- Prevents false positives (pots, wall patterns) from affecting your scaling

**Code**:
```python
# In magic_mirror.py (visualization only)
confidence = detection.confidence
if confidence < 0.75:
    continue  # Skip low-confidence detections

# In roi_from_face.py (AFFECTS SCALING!)
high_conf_detections = [
    det for det in detections.detections 
    if hasattr(det, 'confidence') and det.confidence >= 0.75
]
```

---

### 2. **ROI Selection for Distance Measurement** (CRITICAL)
**Location**: `utils/roi_from_face.py` lines 42-45

**What happens**:
- From all HIGH-CONFIDENCE detections, picks the **LARGEST** face by pixel area
- This detection's bounding box becomes the ROI for depth measurement
- **BEFORE FIX**: Was picking largest of ALL detections (including 60% confidence pots!)
- **AFTER FIX**: Only picks from 75%+ confidence detections

**Code**:
```python
def rect_area(det: ImgDetectionExtended) -> float:
    return det.rotated_rect.size.width * det.rotated_rect.size.height

best_det = max(high_conf_detections, key=rect_area)  # Largest face wins
```

---

### 3. **Distance Measurement** (CRITICAL)
**Location**: `magic_mirror.py` lines 314-337

**What happens**:
- `MeasureDistance` node measures depth (Z distance) at the ROI from step 2
- Raw depth reading: `z_mm = distance_msg.spatials.z`
- **Moving average filter** smooths over last 3 readings: `depth_history.append(z_mm)`
- **Clamped** to valid range: `max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, z_mm))`

**Code**:
```python
z_mm = distance_msg.spatials.z

# Smooth with moving average (last 3 frames)
depth_history.append(z_mm)
z_mm = sum(depth_history) / len(depth_history)

# Clamp to 200-2000mm range
z_mm = max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, z_mm))
```

**Constants**:
```python
MIN_DISTANCE_MM = 200    # Closest valid distance
MAX_DISTANCE_MM = 2000   # Furthest valid distance
REFERENCE_DISTANCE_MM = 500  # Distance where scale = 1.0
```

---

### 4. **Scale Calculation** (CRITICAL)
**Location**: `magic_mirror.py` lines 334-337

**What happens**:
- Inverse relationship: closer = bigger, farther = smaller
- At reference distance (500mm): scale = 1.0
- At 300mm (closer): scale = 1.67 (67% bigger)
- At 1000mm (farther): scale = 0.5 (50% smaller)

**Formula**:
```python
target_scale = REFERENCE_DISTANCE_MM / z_mm
# Example: 500mm / 300mm = 1.67
# Example: 500mm / 1000mm = 0.5

# Clamp to prevent extreme scaling
target_scale = max(MIN_SCALE, min(MAX_SCALE, target_scale))
```

**Constants**:
```python
MIN_SCALE = 0.2   # Maximum zoom out (20% of normal)
MAX_SCALE = 3.0   # Maximum zoom in (300% of normal)
```

---

### 5. **Scale Smoothing** (PREVENTS JITTER)
**Location**: `magic_mirror.py` line 350

**What happens**:
- Gradual transition between old and new scale values
- Prevents sudden jumps when distance changes
- 50% smoothing factor = takes 2-3 frames to reach target

**Code**:
```python
SMOOTHING_FACTOR = 0.5
current_scale += (target_scale - current_scale) * SMOOTHING_FACTOR
```

**Effect**: If target jumps from 1.0 to 2.0:
- Frame 1: current = 1.0 + (2.0 - 1.0) × 0.5 = 1.5
- Frame 2: current = 1.5 + (2.0 - 1.5) × 0.5 = 1.75
- Frame 3: current = 1.75 + (2.0 - 1.75) × 0.5 = 1.875
- Eventually converges to 2.0

---

### 6. **Base Display Scaling** (ALWAYS APPLIED)
**Location**: `magic_mirror.py` lines 353-356

**What happens**:
- Camera: 640×480 pixels (landscape)
- Display: 2160×3840 pixels (portrait)
- Base scale fills display with camera feed at reference distance

**Code**:
```python
base_scale_x = DISPLAY_WIDTH / orig_w   # 2160 / 640 = 3.375
base_scale_y = DISPLAY_HEIGHT / orig_h  # 3840 / 480 = 8.0
base_scale = max(base_scale_x, base_scale_y)  # = 8.0 (use larger to fill)
```

---

### 7. **Combined Scaling** (FINAL IMAGE SIZE)
**Location**: `magic_mirror.py` lines 359-363

**What happens**:
- Combines base display scaling with distance-based scaling
- This is THE FINAL scale applied to your image

**Formula**:
```python
combined_scale = base_scale × current_scale
# At reference distance: 8.0 × 1.0 = 8.0
# When closer (300mm): 8.0 × 1.67 = 13.36
# When farther (1000mm): 8.0 × 0.5 = 4.0

final_w = int(orig_w * combined_scale)  # 640 × combined_scale
final_h = int(orig_h * combined_scale)  # 480 × combined_scale
frame = cv2.resize(frame, (final_w, final_h))
```

---

## Summary: How Scaling Works End-to-End

1. **YuNet detects faces** → Returns detections with confidence scores
2. **Filter by confidence** → Only keep detections ≥ 75% (removes false positives)
3. **Pick largest face** → `ROIFromFace` selects biggest HIGH-CONFIDENCE face
4. **Measure distance** → Depth camera measures Z distance at face ROI
5. **Smooth depth** → Moving average over last 3 frames
6. **Calculate scale** → `target_scale = 500 / z_mm` (inverse proportional)
7. **Smooth scale** → Gradually transition to target (prevents jitter)
8. **Apply combined scale** → `base_scale × current_scale` = final image size
9. **Resize frame** → OpenCV resizes to final dimensions

---

## Key Insight: The Fix

**BEFORE**: 
- ROI was selected from ALL detections (including 60% confidence pots)
- Large false positives could be picked as "largest face"
- Their distance measurement would throw off your scaling

**AFTER**:
- ROI only selected from 75%+ confidence detections
- False positives filtered out before affecting distance measurement
- Only real faces control your scaling

---

## Variables That Affect Your Scaling

### Critical Variables:
- `z_mm` - Distance to face in millimeters (from depth camera)
- `target_scale` - Calculated scale based on distance
- `current_scale` - Smoothed actual scale (what's currently applied)
- `combined_scale` - Final scale (base × current)

### Constants You Can Tune:
- `REFERENCE_DISTANCE_MM = 500` - Distance where image is 1:1 scale
- `MIN_DISTANCE_MM = 200` - Closest valid measurement
- `MAX_DISTANCE_MM = 2000` - Furthest valid measurement
- `MIN_SCALE = 0.2` - Minimum zoom (furthest away)
- `MAX_SCALE = 3.0` - Maximum zoom (closest)
- `SMOOTHING_FACTOR = 0.5` - How fast scale changes (0-1)
- **Confidence threshold = 0.75** - Minimum face detection confidence

### Filters Applied:
1. Confidence filter: `if confidence < 0.75: continue`
2. Distance clamp: `max(200, min(2000, z_mm))`
3. Scale clamp: `max(0.2, min(3.0, target_scale))`
4. Moving average: Last 3 depth readings averaged
5. Exponential smoothing: 50% interpolation per frame
