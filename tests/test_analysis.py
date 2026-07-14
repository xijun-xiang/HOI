from hoi.analysis import compare


def test_scale_aware_comparison():
    baseline = {
        "evaluation": [
            {"task": "door", "mean_return": 10.0},
            {"task": "package", "mean_return": -100.0},
        ]
    }
    candidate = {
        "evaluation": [
            {"task": "door", "mean_return": 15.0},
            {"task": "package", "mean_return": -110.0},
        ]
    }
    result = compare(baseline, candidate)
    assert result["raw_delta_mean"] == -2.5
    assert result["normalised_delta_mean"] == 0.2
