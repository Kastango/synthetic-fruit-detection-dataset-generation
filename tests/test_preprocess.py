from fruit_pipeline.preprocess import resolve_torch_device


def test_numeric_yolo_device_is_accepted_by_torch() -> None:
    assert resolve_torch_device(
        "0", cuda_available=True, mps_available=False
    ) == "cuda:0"


def test_automatic_device_prefers_available_accelerator() -> None:
    assert resolve_torch_device(
        "auto", cuda_available=False, mps_available=True
    ) == "mps"
    assert resolve_torch_device(
        None, cuda_available=False, mps_available=False
    ) == "cpu"
