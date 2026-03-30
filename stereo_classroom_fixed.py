import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse
import os


# ------------------------------------------------------------
# Image alignment for phone-captured stereo pair
# ------------------------------------------------------------

def align_right_to_left(left_img, right_img):
    """
    Align the right image to the left image using ORB feature matching.
    This is a simple practical substitute for full stereo rectification.
    """
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("Warning: Not enough features found for alignment. Using original right image.")
        return right_img

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        print("Warning: Not enough matches found for alignment. Using original right image.")
        return right_img

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:200]

    pts_left = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)

    if H is None:
        print("Warning: Homography failed. Using original right image.")
        return right_img

    aligned_right = cv2.warpPerspective(
        right_img,
        H,
        (left_img.shape[1], left_img.shape[0])
    )

    return aligned_right


# ------------------------------------------------------------
# Disparity computation
# ------------------------------------------------------------

def compute_disparity(left_gray, right_gray):
    """
    Compute disparity map using StereoSGBM.
    Output disparity is float32 in pixel units.
    """
    min_disp = 0
    num_disp = 16 * 8   # must be multiple of 16
    block_size = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def save_disparity_visual(disparity, save_path):
    """
    Save a viewable grayscale disparity image.
    """
    disp_vis = disparity.copy()
    valid = disp_vis > 0

    out = np.zeros_like(disp_vis, dtype=np.uint8)
    if np.any(valid):
        min_val = np.min(disp_vis[valid])
        max_val = np.max(disp_vis[valid])
        out[valid] = ((disp_vis[valid] - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)

    cv2.imwrite(save_path, out)


def get_valid_disparity(disparity, x, y, window=7):
    """
    Get robust disparity near (x, y) using median of valid disparities in a local window.
    """
    h, w = disparity.shape

    x1 = max(0, x - window)
    x2 = min(w, x + window + 1)
    y1 = max(0, y - window)
    y2 = min(h, y + window + 1)

    patch = disparity[y1:y2, x1:x2]
    valid = patch[patch > 0.5]

    if len(valid) == 0:
        return None

    return float(np.median(valid))


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

def pixel_to_3d(u, v, d, fx, fy, cx, cy, baseline):
    """
    Convert pixel location and disparity to 3D camera coordinates.
    Units match the baseline unit, so use meters for baseline.
    """
    Z = (fx * baseline) / d
    X = ((u - cx) * Z) / fx
    Y = ((v - cy) * Z) / fy
    return X, Y, Z


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

def detect_objects(left_img, model):
    """
    Detect chairs and tables.
    COCO ids:
      chair = 56
      dining table = 60
    """
    results = model(left_img, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if cls_id not in [56, 60]:
            continue

        if conf < 0.25:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        label = "chair" if cls_id == 56 else "table"

        detections.append({
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })

    return detections


# ------------------------------------------------------------
# Position estimation
# ------------------------------------------------------------

def estimate_floor_position(det, disparity, fx, fy, cx, cy, baseline):
    x1, y1, x2, y2 = det["bbox"]

    candidate_points = [
        (int((x1 + x2) / 2), int(y2)),
        (int((x1 + x2) / 2), int(y2 - 5)),
        (int((x1 + x2) / 2), int(y2 - 10)),
        (int(x1 + 0.4 * (x2 - x1)), int(y2 - 5)),
        (int(x1 + 0.6 * (x2 - x1)), int(y2 - 5)),
    ]

    disparities = []
    chosen_u, chosen_v = None, None

    for u, v in candidate_points:
        d = get_valid_disparity(disparity, u, v, window=7)
        if d is not None:
            disparities.append((d, u, v))

    if not disparities:
        return None

    disparities.sort(key=lambda t: t[0])
    d, chosen_u, chosen_v = disparities[len(disparities) // 2]

    X, Y, Z = pixel_to_3d(chosen_u, chosen_v, d, fx, fy, cx, cy, baseline)

    return {
        "label": det["label"],
        "u": chosen_u,
        "v": chosen_v,
        "disparity": d,
        "X_floor": X,
        "Y_floor": Z
    }


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def draw_detections(img, detections, positions):
    out = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]

        color = (255, 0, 0) if label == "chair" else (0, 0, 255)  # chair blue, table red in BGR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        u = int((x1 + x2) / 2)
        v = int(y2)
        cv2.circle(out, (u, v), 5, (0, 255, 255), -1)

        text = label
        for p in positions:
            if p["label"] == label and p["u"] == u and p["v"] == v:
                text += f" X={p['X_floor']:.2f}, Y={p['Y_floor']:.2f}"
                break

        cv2.putText(
            out,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return out


def plot_topdown(positions, save_path):
    plt.figure(figsize=(8, 6))

    table_added = False
    chair_added = False

    for p in positions:
        if p["label"] == "table":
            plt.scatter(
                p["X_floor"],
                p["Y_floor"],
                c="red",
                s=100,
                label="table" if not table_added else None
            )
            plt.text(p["X_floor"], p["Y_floor"], "table", fontsize=9)
            table_added = True
        elif p["label"] == "chair":
            plt.scatter(
                p["X_floor"],
                p["Y_floor"],
                c="blue",
                s=100,
                label="chair" if not chair_added else None
            )
            plt.text(p["X_floor"], p["Y_floor"], "chair", fontsize=9)
            chair_added = True

    plt.xlabel("X position (meters)")
    plt.ylabel("Y position / depth (meters)")
    plt.title("Top-Down 2D Plot of Tables and Chairs")
    plt.grid(True)

    if table_added or chair_added:
        plt.legend()

    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=str, required=True, help="Path to left image")
    parser.add_argument("--right", type=str, required=True, help="Path to right image")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)

    if left_img is None or right_img is None:
        raise FileNotFoundError("Could not load left or right image.")

    # Resize right image if sizes differ
    if left_img.shape[:2] != right_img.shape[:2]:
        right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))

    # Align right image to left image
    aligned_right = align_right_to_left(left_img, right_img)
    cv2.imwrite(os.path.join(args.output_dir, "aligned_right.png"), aligned_right)

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(aligned_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = compute_disparity(left_gray, right_gray)
    save_disparity_visual(disparity, os.path.join(args.output_dir, "disparity.png"))

    # Approximate phone/stereo parameters
    # You can slightly adjust baseline if needed based on how far you moved the phone
    fx = 700.0
    fy = 700.0
    cx = left_img.shape[1] / 2.0
    cy = left_img.shape[0] / 2.0
    baseline = 0.08   # meters, about 8 cm horizontal phone shift

    # Load detector
    model = YOLO("yolov8n.pt")

    # Detect objects
    detections = detect_objects(left_img, model)
    print(f"Detected objects: {len(detections)}")

    # Estimate positions
    positions = []
    for det in detections:
        pos = estimate_floor_position(det, disparity, fx, fy, cx, cy, baseline)
        if pos is not None:
            positions.append(pos)

    print(f"Objects with valid positions: {len(positions)}")

    for p in positions:
        print(f"{p['label']} -> X = {p['X_floor']:.2f} m, Y = {p['Y_floor']:.2f} m")

    # Save annotated detections
    annotated = draw_detections(left_img, detections, positions)
    cv2.imwrite(os.path.join(args.output_dir, "detections_and_positions.png"), annotated)

    # Save final top-down plot
    plot_topdown(positions, os.path.join(args.output_dir, "topdown_plot.png"))

    print(f"Saved outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()