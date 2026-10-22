from typing import Any, Dict


def get_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute the intersection-over-union of two character ranges."""
    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union <= 0:
        return 0
    return overlap / union


def is_found(correct: Dict[str, Any], retrieved: Dict[str, Any]) -> bool:
    """Return True if a retrieved source matches the correct one."""
    if correct["file_path"] != retrieved["file_path"]:
        return False
    iou = get_iou(correct["first_character_index"],
                  correct["last_character_index"],
                  retrieved["first_character_index"],
                  retrieved["last_character_index"])
    return iou >= 0.05
