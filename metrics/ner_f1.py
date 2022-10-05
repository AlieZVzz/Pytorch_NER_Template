from utils.utils import get_tags


def f1_score(tar_path, pre_path, tag, tag_map):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1
