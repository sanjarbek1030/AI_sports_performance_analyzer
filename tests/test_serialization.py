import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sports_analyzer.config import Config
from sports_analyzer.models import Event, PossessionSpell
from sports_analyzer.reporting.report_generator import ReportGenerator, to_jsonable
from sports_analyzer.utils import format_timestamp


def test_to_jsonable_handles_numpy_nan_and_dataclasses():
    spell = PossessionSpell(1, 0, 0, 10, 0.0, 0.4, 0.44, 0.9, (1.0, 2.0), None)
    payload = {"a": np.float64(1.5), "b": np.int64(3), "c": float("nan"), "d": np.array([1.0, math.inf]),
               "e": spell, "f": (1, 2), "g": np.bool_(True), "h": Path("x/y"), "i": pd.DataFrame({"k": [1, None]})}
    text = json.dumps(to_jsonable(payload), allow_nan=False)          # strict JSON: no NaN/Infinity tokens
    loaded = json.loads(text)
    assert loaded["a"] == 1.5 and loaded["c"] is None and loaded["d"] == [1.0, None]
    assert loaded["e"]["player_id"] == 1 and loaded["g"] is True and loaded["f"] == [1, 2]


def test_event_log_csv_format():
    events = [Event("pass", 13.42, 335, 7, 0, 0.8123, {"successful": True}), Event("shot", 41.11, 1028, 10, 1, 0.7, {})]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "event_log.csv"
        ReportGenerator(Config()).write_event_log(events, path)
        lines = path.read_text().splitlines()
    assert lines[0] == "timestamp,frame,event_type,player_id,team_id,confidence,metadata"
    assert lines[1].startswith("00:13.42,335,PASS,7,Team A,0.812,") and lines[2].startswith("00:41.11,1028,SHOT,10,Team B")


def test_dataframe_csv_writes_none_as_empty_cell():
    df = pd.DataFrame([{"player_id": 1, "distance_km": None, "passes": 3}], dtype=object)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.csv"
        ReportGenerator.write_dataframe(df, path)
        assert path.read_text().splitlines()[1] == "1,,3"


def test_format_timestamp():
    assert format_timestamp(13.42) == "00:13.42" and format_timestamp(125.5) == "02:05.50" and format_timestamp(-1) == "00:00.00"
