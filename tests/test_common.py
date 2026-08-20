from fruit_pipeline.common import automatic_workers


def test_automatic_workers_is_conservative() -> None:
    assert automatic_workers(1) == 1
    assert automatic_workers(4) == 2
    assert automatic_workers(16) == 8
    assert automatic_workers(128) == 8
