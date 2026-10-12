#!/usr/bin/env python3
"""Interactive calibration: click known pitch landmarks on one video frame.

    python scripts/select_calibration_points.py input/sports_input.mp4 --frame 0 \
        --out config/calibration.json

Workflow
  1. A frame opens. Left-click a landmark you can identify precisely
     (corner, penalty-box corner, centre spot ...).
  2. The terminal asks which landmark it is (type the name, or a number from
     the printed list). Repeat for >= 4 points spread over the pitch
     (more points = more robust; avoid points on a single line).
  3. Press ENTER in the image window (or 'q') to finish. The homography is
     validated and the JSON is written.
Right-click removes the last point. Requires a desktop OpenCV window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sports_analyzer.config import Config  # noqa: E402
from sports_analyzer.exceptions import CalibrationError  # noqa: E402
from sports_analyzer.field.calibration import compute_homography  # noqa: E402
from sports_analyzer.field.football_field import FootballField  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="config/calibration.json")
    parser.add_argument("--length", type=float, default=105.0)
    parser.add_argument("--width", type=float, default=68.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Cannot read frame {args.frame} of {args.video}")
        return 1

    field = FootballField(args.length, args.width)
    names = sorted(field.landmarks())
    for i, name in enumerate(names):
        print(f"{i:2d}  {name}")

    points: list[tuple[int, int]] = []
    labels: list[str] = []
    window = "Click landmarks (ENTER=finish, right-click=undo)"

    def redraw() -> None:
        canvas = frame.copy()
        for (x, y), label in zip(points, labels):
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(canvas, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            label = input(f"\nPoint at ({x}, {y}) - landmark name or number: ").strip()
            name = names[int(label)] if label.isdigit() and int(label) < len(names) else label
            if name not in field.landmarks():
                print(f"Unknown landmark '{label}', point ignored.")
                return
            points.append((x, y)); labels.append(name)
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop(); labels.pop()
        redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while cv2.waitKey(50) & 0xFF not in (13, ord("q")):
        pass
    cv2.destroyAllWindows()

    try:
        compute_homography(points, [field.landmarks()[n] for n in labels], Config().field.max_reprojection_error_m)
    except CalibrationError as exc:
        print(f"Calibration rejected: {exc}")
        return 1
    h, w = frame.shape[:2]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"frame_size": [w, h], "image_points": [list(p) for p in points],
                                          "landmarks": labels}, indent=2))
    print(f"Calibration saved to {args.out}. Use it with: python main.py --calibration {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
