def check_args(query, k):
    if not isinstance(k, int) or k <= 0:
        print("Invalid Argument: --k must be a positive integer")
        return False
    if not query.strip():
        print("Invalid Argument: query must not be empty")
        return False
    return True


def get_iou(start1, end1, start2, end2):
    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union <= 0:
        return 0
    return overlap / union

def is_found(correct, retrieved):
    if correct["file_path"] != retrieved["file_path"]:
        return False
    iou = get_iou(correct["first_character_index"],
                  correct["last_character_index"],
                  retrieved["first_character_index"],
                  retrieved["last_character_index"])
    return iou >= 0.05


