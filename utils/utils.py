import numpy as np
import torch


def pad(batch):
    """Pads to the longest sample"""
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = max(np.array(seqlens))

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens


def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("M-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S-" + tag)
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    for index, tag in enumerate(path):
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == o_tag:
            begin = -1
        elif tag == single_tag:
            tags.append([index, index])
        last_tag = tag
    return tags


def format_result(output, index, result, text, tag):
    for i in result:
        begin, end = i
        entity_dict = output[index]['entities']
        entity_dict.append({
            "start": begin - 1,
            "stop": end,
            "entity": ''.join(text[begin:end + 1]),
            "type": tag
        })

        # print(text[begin:end+1])
        # output[index]['entities']['start'] = begin
        # output[index]['entities']['stop'] = end + 1
        # output[index]['entities']['idx'] = text[begin:end + 1]
        # output[index]['entities']['stop'] = tag

    return output


