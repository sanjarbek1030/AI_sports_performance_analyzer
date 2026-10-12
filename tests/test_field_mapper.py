import cv2
import numpy as np

from helpers import raises
from sports_analyzer.exceptions import CalibrationError
from sports_analyzer.field.calibration import ManualCalibration, NullCalibration, compute_homography
from sports_analyzer.field.field_mapper import FieldMapper
from sports_analyzer.field.football_field import FootballField

FIELD = FootballField()
NAMES = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left", "center_spot"]
FIELD_PTS = [FIELD.landmarks()[n] for n in NAMES]
H_TRUE = np.array([[12, 2, 100], [0.5, 7, 150], [0.0005, 0.002, 1]], float)  # field -> image (perspective)


def _image_points():
    return cv2.perspectiveTransform(np.array(FIELD_PTS, float).reshape(-1, 1, 2), H_TRUE).reshape(-1, 2)


def test_perspective_round_trip():
    mapper = FieldMapper(FIELD, ManualCalibration(_image_points(), FIELD_PTS, 1.0))
    img = cv2.perspectiveTransform(np.array([[30.0, 20.0]]).reshape(-1, 1, 2), H_TRUE).reshape(-1, 2)
    x, y = mapper.to_field_point(*img[0])
    assert abs(x - 30) < 1e-3 and abs(y - 20) < 1e-3
    back = mapper.to_image([[30.0, 20.0]])
    assert np.allclose(back, img, atol=1e-2)


def test_uncalibrated_mapper_returns_none():
    mapper = FieldMapper(FIELD, NullCalibration())
    assert not mapper.is_calibrated and mapper.to_field([[1, 2]]) is None


def test_invalid_calibrations_are_rejected():
    ip = _image_points()
    with raises(CalibrationError):
        compute_homography(ip[:3], FIELD_PTS[:3], 1.0)               # too few points
    with raises(CalibrationError):
        compute_homography([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 0], [2, 0], [3, 1]], 1.0)  # collinear
    wrong = list(FIELD_PTS)
    wrong[4] = (10.0, 60.0)                                           # mismatched correspondence
    with raises(CalibrationError):
        compute_homography(ip, wrong, 0.5)


def test_on_pitch_margin():
    mapper = FieldMapper(FIELD, NullCalibration())
    assert mapper.on_pitch((52, 34), 0) and not mapper.on_pitch((120, 34), 2) and mapper.on_pitch((106, 34), 2)


def test_landmarks_follow_field_dimensions():
    small = FootballField(100, 60)
    assert small.landmarks()["corner_bottom_right"] == (100.0, 60.0)
    assert small.goal_mouth(+1) == (100.0, 30 - 3.66, 30 + 3.66)
