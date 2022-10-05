
file = open("../data/训练集/train.txt", "r", encoding="utf-8")
train_file = open("data/processed_bmeo_train.txt", "w", encoding="utf-8")
dev_file = open("data/processed_bmeo_dev.txt", "w", encoding="utf-8")
sentence = 0
for i in file.readlines():
    if i == '\n':
        sentence = sentence + 1
    if sentence < 1000:
        train_file.write(i)
    elif sentence > 1000:
        dev_file.write(i)
#     if i == '\n':
#         sentence += 1
# print(sentence)
file.close()
train_file.close()
dev_file.close()