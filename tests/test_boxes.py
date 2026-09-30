import numpy as np

from fruit_pipeline.boxes import iou_matrix, match_counts, read_boxes


def test_read_boxes_converts_yolo_to_pixels(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n\n0 0.1 0.2 0.2 0.2\n")
    boxes = read_boxes(label, 100, 50)
    np.testing.assert_allclose(boxes, [[40, 15, 60, 35], [0, 5, 20, 15]])
    label.write_text("")
    assert read_boxes(label, 100, 50).shape == (0, 4)


def test_iou_matrix_handles_overlap_and_empty():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[50 / 150, 0]])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_counts_is_one_to_one_by_confidence():
    truth = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    # Duas previsões disputam a primeira caixa; só a mais confiante acerta.
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [80, 80, 90, 90]], dtype=float)
    scores = np.array([0.4, 0.9, 0.1])
    assert match_counts(truth, boxes, scores, 0.25) == (1, 1, 1)
    assert match_counts(truth, boxes, scores, 0.0) == (1, 2, 1)
    assert match_counts(truth, boxes, scores, 0.95) == (0, 0, 2)
