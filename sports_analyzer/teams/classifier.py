"""Team classification from jersey appearance.

Problem solved : decide which team each player belongs to without hard-coded
                 colours.
Pipeline       : player crop -> upper-body region -> remove grass (HSV) ->
                 dominant colour (k-means, Lab) -> one 3-D feature per crop.
                 Features from the whole video are clustered (robust,
                 outlier-trimmed 2-means); each track then takes a confidence-
                 weighted vote over all its crops.
Outputs        : per-track ``team_id`` (0, 1, or -1 = unknown e.g. goalkeepers
                 / referees whose kit matches neither cluster) + confidence.
Replaceable    : implement ``TeamClassifier`` (extract / fit / predict) to plug
                 in a trained CNN later; the pipeline only uses that interface.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

from ..config import TeamConfig
from ..exceptions import InsufficientDataError
from ..models import BBox, DetectionRole, PlayerTrack, TEAM_REFEREE, TEAM_UNKNOWN

logger = logging.getLogger(__name__)
_LAB_L_SCALE = 100.0 / 255.0
_LAB_OFFSET = 128.0
_PIXEL_SUBSAMPLE = 300
_KMEANS_MIN_PIXELS = 40
_EPS = 1e-9


@dataclass
class TeamPrediction:
    team_id: int
    confidence: float


class TeamClassifier(ABC):
    @abstractmethod
    def extract(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Return a feature vector for one player crop (or None if unusable)."""

    @abstractmethod
    def fit(self, features: np.ndarray) -> None:
        """Learn team structure from many features (unsupervised or otherwise)."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> list[TeamPrediction]:
        """Classify each feature row."""


class ColorKMeansTeamClassifier(TeamClassifier):
    def __init__(self, cfg: TeamConfig) -> None:
        self.cfg = cfg
        self.centroids: Optional[np.ndarray] = None
        self.spread: Optional[np.ndarray] = None
        cv2.setRNGSeed(cfg.random_seed)

    # ----------------------------------------------------------- features
    def extract(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = y2 - y1, x2 - x1
        if h < 8 or w < 4:
            return None
        c = self.cfg
        top, bottom = y1 + int(c.upper_body_top * h), y1 + int(c.upper_body_bottom * h)
        left, right = x1 + int(c.horizontal_margin * w), x2 - int(c.horizontal_margin * w)
        top, left = max(top, 0), max(left, 0)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass = (hsv[:, 0] >= c.grass_hue_low) & (hsv[:, 0] <= c.grass_hue_high) & (hsv[:, 1] >= c.grass_min_saturation)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)[~grass]
        if len(lab) < c.min_jersey_pixels:
            return None
        if len(lab) > _PIXEL_SUBSAMPLE:
            lab = lab[np.linspace(0, len(lab) - 1, _PIXEL_SUBSAMPLE).astype(int)]
        color = self._dominant_color(lab)
        return np.array([
            c.lightness_weight * color[0] * _LAB_L_SCALE,
            color[1] - _LAB_OFFSET,
            color[2] - _LAB_OFFSET,
        ], dtype=np.float32)

    @staticmethod
    def _dominant_color(lab: np.ndarray) -> np.ndarray:
        if len(lab) < _KMEANS_MIN_PIXELS:
            return np.median(lab, axis=0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(lab, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=2)
        return centers[int(np.argmax(counts))]

    # ---------------------------------------------------------------- fit
    def fit(self, features: np.ndarray) -> None:
        c = self.cfg
        data = np.asarray(features, dtype=float)
        if len(data) < c.min_fit_samples:
            raise InsufficientDataError(
                f"Only {len(data)} jersey samples (need {c.min_fit_samples}) to learn team colours.")
        rng = np.random.default_rng(c.random_seed)
        if len(data) > c.fit_max_samples:
            data = data[rng.choice(len(data), c.fit_max_samples, replace=False)]
        centroids, _ = kmeans2(data, 2, minit="++", seed=rng)
        for _ in range(c.trim_iterations):  # trim outliers (referees, GKs) and refit
            dist = np.linalg.norm(data[:, None] - centroids[None], axis=2).min(axis=1)
            keep = dist <= np.quantile(dist, 1.0 - c.trim_fraction)
            centroids, _ = kmeans2(data[keep], centroids, minit="matrix")
        dist_all = np.linalg.norm(data[:, None] - centroids[None], axis=2)
        nearest = dist_all.argmin(axis=1)
        spread = np.array([
            np.quantile(dist_all[nearest == k, k], 0.9) if np.any(nearest == k) else 1.0 for k in range(2)
        ])
        order = np.argsort(-centroids[:, 0])  # team 0 = lighter kit (deterministic naming)
        self.centroids, self.spread = centroids[order], np.maximum(spread[order], 1.0)
        logger.info("Team colour centroids (Lab-like): A=%s B=%s", np.round(self.centroids[0], 1), np.round(self.centroids[1], 1))

    # ------------------------------------------------------------ predict
    def predict(self, features: np.ndarray) -> list[TeamPrediction]:
        if self.centroids is None:
            raise InsufficientDataError("TeamClassifier.fit() has not been called.")
        feats = np.atleast_2d(np.asarray(features, dtype=float))
        dist = np.linalg.norm(feats[:, None] - self.centroids[None], axis=2)
        out: list[TeamPrediction] = []
        for row in dist:
            k = int(np.argmin(row))
            if row[k] > self.cfg.outlier_scale * self.spread[k]:
                out.append(TeamPrediction(TEAM_UNKNOWN, 0.0))  # matches neither kit
                continue
            other = row[1 - k]
            out.append(TeamPrediction(k, float((other - row[k]) / (other + row[k] + _EPS))))
        return out


def assign_track_teams(classifier: TeamClassifier, tracks: dict[int, PlayerTrack], cfg: TeamConfig) -> None:
    """Confidence-weighted vote of every crop of a track -> track team (in place)."""
    for track in tracks.values():
        if track.role == DetectionRole.REFEREE:
            track.team_id, track.team_confidence = TEAM_REFEREE, 1.0
        else:
            samples = [o for o in track.observations if o.jersey_feature is not None]
            track.team_id, track.team_confidence = TEAM_UNKNOWN, 0.0
            if len(samples) >= cfg.min_votes:
                preds = classifier.predict(np.stack([o.jersey_feature for o in samples]))
                weights = np.zeros(3)  # [team0, team1, unknown]
                for pred in preds:
                    weights[2 if pred.team_id == TEAM_UNKNOWN else pred.team_id] += max(pred.confidence, 0.05)
                best = int(np.argmax(weights))
                share = weights[best] / weights.sum()
                if best != 2 and share >= cfg.min_vote_share:
                    mean_conf = float(np.mean([p.confidence for p in preds if p.team_id == best]))
                    track.team_id, track.team_confidence = best, float(share * mean_conf)
        for obs in track.observations:
            obs.team_id = track.team_id
