from fruit_pipeline.common import ROOT, load_yaml
from fruit_pipeline.training import expand_experiments


def test_baseline_grid_is_test_blind_and_has_no_augmentation() -> None:
    config = load_yaml(ROOT / "configs" / "experiments.yaml")
    specs = expand_experiments(config, allow_missing=True)
    assert len(specs) == 24  # 2 condições piloto × 2 modelos × 2 LR × 3 sementes
    assert all("test" not in spec for spec in specs)
    assert all(spec["parameters"]["mosaic"] == 0 for spec in specs)
    assert all(spec["parameters"]["hsv_s"] == 0 for spec in specs)
    assert all(spec["parameters"]["fliplr"] == 0 for spec in specs)
