"""Group per-frame observations into per-player ``PlayerTrack`` objects."""
from __future__ import annotations

from ..models import FrameData, PlayerTrack


def group_observations_into_tracks(frames: list[FrameData]) -> dict[int, PlayerTrack]:
    """Observations are shared (not copied) between ``FrameData`` and the tracks,
    so analysers that fill ``smoothed_xy`` / ``speed_kmh`` update both views."""
    tracks: dict[int, PlayerTrack] = {}
    for frame in frames:
        for obs in frame.players:
            track = tracks.get(obs.player_id)
            if track is None:
                track = tracks[obs.player_id] = PlayerTrack(player_id=obs.player_id, role=obs.role)
            track.observations.append(obs)
    return tracks
