# Component guide

For every component: **Problem** (what it solves), **Why** (why it is necessary), **In** (data it
receives), **Out** (data it produces), **Interacts** (who it talks to).

---
## VideoProcessor (`video/processor.py`)
* **Problem:** OpenCV fails quietly - zero FPS, unreadable frames, writers that never open.
* **Why:** every timestamp (and therefore every speed) is `frame / fps`; wrong FPS corrupts all statistics. The output must keep the input FPS/resolution.
* **In:** input path. **Out:** `VideoInfo` (width, height, FPS, frame count, duration), frame/batch iterators, a validated writer.
* **Interacts:** the pipeline reads the video twice (analysis pass, render pass). Missing/corrupt/unsupported files raise `VideoOpenError`; mid-stream failures are skipped with warnings (or the previous frame is repeated in the output so frame count is preserved) and abort after N consecutive failures; unopenable writers raise `VideoWriteError` after trying fallback codecs.

## PlayerDetector (`detection/player_detector.py`)
* **Problem:** find players (and goalkeepers/referees) in each frame.
* **Why:** everything starts from boxes. The *feet* point `(x1+x2)/2, y2` is stored because only ground contact can be mapped to the pitch; a box centre floats in the air and biases positions by up to a metre.
* **In:** batches of frames. **Out:** `Detection` (bbox in original pixels, confidence, class, frame, timestamp). Runs at a low confidence floor so the tracker can use weak boxes.
* **Interacts:** feeds PlayerTracker. `video.process_width` shrinks only the inference input; boxes are scaled back.

## PlayerTracker (`tracking/player_tracker.py`, `iou_tracker.py`)
* **Problem:** same player = same id across frames, despite detection order changes and brief occlusion.
* **Why:** movement statistics are per identity.
* **In:** detections + frame. **Out:** detections with `track_id`. Ultralytics ByteTrack/BoT-SORT are driven directly with our detections (no second inference); `track_buffer_s` keeps lost tracks alive. If they cannot start, a built-in two-stage IoU tracker takes over (logged).
* **Interacts:** PlayerIdentityManager consumes its output.

## PlayerIdentityManager (`tracking/identity.py`)
* **Problem:** trackers forget after the buffer expires; the returning player gets a new id.
* **Why:** avoids fragmenting one player into many "players".
* **In:** tracked detections + jersey colour features. **Out:** stable `player_id`. A new tracker id is linked to a recently lost identity only if the gap is short, the reappearance is physically reachable (distance in player-heights vs elapsed time) and jersey colour matches; one-to-one greedy matching.
* **Interacts:** after PlayerTracker, before field mapping. No ReID embeddings yet.

## BallDetector (`detection/ball_detector.py`)
* **Problem:** the ball is tiny, blurred, often hidden.
* **Why:** needs separate settings (native resolution, larger `imgsz`, low confidence) and shape filters; picking the "best" box per frame here would hard-code mistakes.
* **In:** full-resolution frames. **Out:** up to N `BallDetection` candidates per frame. **Interacts:** BallTracker chooses among candidates.

## BallTracker (`tracking/ball_tracker.py`)
* **Problem:** detections flicker; false positives (white boots, line marks) appear.
* **Why:** possession, passes and shots need a trajectory, not isolated points.
* **In:** candidates per frame. **Out:** `BallState` (position, velocity, direction, confidence, `predicted` flag).
* **How:** constant-velocity Kalman filter; candidates must fall inside a gate around the prediction (impossible jumps rejected); a new track needs N consistent detections; on misses it coasts with decaying confidence for `max_coast_s`, then declares the ball lost (logged) instead of resetting immediately.
* **Interacts:** states are mapped to field coordinates by FieldMapper (using the ball's contact point).

## TeamClassifier (`teams/classifier.py`)
* **Problem:** which team is each player on, without hard-coded RGB.
* **Why:** possession, passes and team statistics need teams.
* **In:** player crops (features extracted during pass 1). **Out:** per-track `team_id` (0, 1, or -1 unknown; referees -2) and confidence.
* **How:** upper-body crop -> mask grass in HSV -> dominant colour (k-means) in Lab with down-weighted lightness -> robust trimmed 2-means over all features -> distance-ratio confidence; outliers (referee, goalkeeper kits) become *Unknown* -> confidence-weighted vote per track. The abstract `TeamClassifier` (`extract/fit/predict`) lets a trained CNN replace it.

## Calibration + FieldMapper (`field/`)
* **Problem:** pixels are not metres; perspective distorts distances.
* **Why:** speed, distance, heatmaps, possession radius and goal geometry all need metres.
* **In:** >= 4 image points + matching pitch landmarks (named or explicit). **Out:** validated homography (`CalibrationProvider`) and `FieldMapper.to_field / to_image / on_pitch`.
* **Checks:** point count, duplicates, collinearity, reprojection error, conditioning -> `CalibrationError`. People off the pitch (bench, crowd) are flagged `on_pitch=False` and excluded.
* **Extensible:** `AutomaticCalibration` is the hook for keypoint/line detection or per-frame homography (panning cameras).

## MovementAnalyzer (`analytics/movement.py`)
* **Problem:** raw foot positions jitter; differentiating them yields impossible speeds.
* **In:** `PlayerTrack` with raw field positions. **Out:** smoothed positions, speed (km/h), acceleration, distance, sprints, speed zones, acceleration/deceleration/stop/direction-change events, robust max speed.
* **How:** split at gaps (no motion invented across them) -> reject teleports -> resample uniformly -> smooth (Savitzky-Golay / moving average) -> differentiate -> discard impossible speeds/accelerations as NaN (never clip) -> max speed from a median-filtered series. A sprint needs speed >= threshold for >= `min_sprint_duration` (short dips merged).
* **Interacts:** writes results onto the shared `PlayerObservation` objects used by possession, events, CSV and rendering.

## PossessionAnalyzer (`analytics/possession.py`)
* **Problem:** nearest-player-to-ball flickers and is wrong while the ball travels.
* **In:** per-frame ball and player field positions. **Out:** `PossessionSpell`s (player, team, start, end, duration, confidence) and per-frame states.
* **How:** `FREE -> CANDIDATE -> POSSESSION -> RELEASE -> FREE`; candidate must stay nearest within the radius for `confirm_s`; the holder keeps the ball inside a larger release radius, survives short dropouts, and loses it to a rival who is clearly closer for `confirm_s` or after the ball leaves for `release_s`. Referees never hold the ball. Team percentages are validated to sum to 100.

## EventEngine + football detectors (`events/`)
* **Problem:** detect pass/shot/goal/... without burying sport rules in the pipeline.
* **In:** `EventContext` (tracks, spells, ball kinematics, geometry, earlier events). **Out:** `Event` (type, timestamp, frame, player, team, confidence, metadata, `inferred`).
* **Detectors (ordered):** possession change -> shot (release + fast ball + heading into the goal cone; outcome *inferred*: goal / on_target / saved_candidate / blocked_candidate / off_target) -> pass (consecutive spells, minimum distance and speed, direction consistent with release velocity, ball stayed in play; failed pass if received by the other team) -> goal (line crossing between the posts or lost beside the line, approach speed, **no re-entry** during `goal_confirm_s`; links shot and assist; confidence <= 0.70) -> tackle/interception candidates -> out of bounds -> sprint. A crashing detector is logged and skipped.

## PlayerStatistics / TeamStatistics (`analytics/statistics.py`)
* **Problem:** aggregate honestly. **In:** tracks, spells, events, `DataQuality`. **Out:** DataFrames for the CSVs.
* **Rules:** uncalibrated -> movement stats `None`; ball seen in < 15% of frames -> ball-event stats `None`; identities shorter than `min_report_track_s` and referees are excluded. `None` becomes an empty CSV cell / JSON null, never 0.

## Tactical, formation, roles (`analytics/tactical.py`, `formation.py`, `roles.py`)
* **Measured:** centroid, width, depth, spread, mean pairwise distance, hull area per team (only when >= `min_players` visible), split by in/out of possession, with coverage reported.
* **Inferred (flagged):** narrow/wide/compact/stretched labels relative to pitch size; formation per time window (10 outfield players sorted by depth and cut into lines, best template wins, softmax confidence); role from average attack-normalised position. These never claim to be ground truth.

## HeatmapGenerator (`analytics/heatmaps.py`)
Time-weighted occupancy (each sample adds the time until the next sample), Gaussian-smoothed, drawn on the pitch. Per player, per team, combined.

## Visualizer + MiniMap (`visualization/`)
Draws player ellipse + id + team + speed, possession marker, fading player/ball trails, ball marker, match clock, live possession bar, recent-event banner and a top-down mini-map in the same field coordinates. Debug mode adds confidences, tracker ids, field coordinates, possession distance, raw ball detection and calibration points. Sizes scale with frame height.

## ReportGenerator (`reporting/report_generator.py`)
Streams `player_tracks.csv`; writes statistics/event CSVs and a strict-JSON `match_report.json` (NaN/Inf -> null, numpy/dataclass aware) containing data-quality flags, warnings and the limitations list.

## AnalysisPipeline (`pipeline.py`) and `run_analysis` (`analysis.py`)
Orchestration only: pass 1 (detect -> track -> identity -> features -> map -> ball), analysis (teams -> movement -> ball kinematics -> possession -> events -> statistics -> tactics), reports, pass 2 (render). `run_analysis` takes plain data, which is why the whole analytics stack is testable with synthetic matches.

## SportsAnalyzer (`sports/`)
Interface for field model, attack-direction rules and event detectors. `FootballAnalyzer` is implemented; add basketball/tennis by subclassing and registering in `sports/__init__.py`.
