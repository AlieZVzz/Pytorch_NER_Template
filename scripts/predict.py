import torch
from tqdm import tqdm
from utils.utils import get_tags, format_result
import json
from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = torch.load('checkpoints/bert_bilstm_crf_params.pth', map_location=device)

test_path = '../data/test_data.txt'
file = open(test_path, 'r', encoding='utf-8')
input = file.readlines()[:10]
file.close()
output = []

test = tokenizer.batch_encode_plus(input, padding=True, max_length=256, return_attention_mask=True, return_token_type_ids=True)
for i, input_str in tqdm(enumerate(input)):
    model.eval()
    tokenize_text = tokenizer.encode(input_str)

    x = torch.tensor(tokenize_text).unsqueeze(dim=0)
    text = [tokenizer.convert_ids_to_tokens(j) for j in x]
    output.append({'text': input_str.strip(), "entities": []})
    x = x.to(device)

    with torch.no_grad():
        _, predict = model(x)
    paths = predict.to('cpu').tolist()

    for tag in TAGS:
        tags = get_tags(paths[0], tag, tag2idx)
        # print(tags)
        output = format_result(output, i, tags, text[0], tag)
        # print(output)

with open("../checkpoints/predict.json", "w+", encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)
