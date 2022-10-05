from sklearn.model_selection import train_test_split

total_list = []
singe_list = []
with open('../data/Denmark.txt.anns', 'r', encoding='utf-8') as f:
    for i in f.readlines():
        # print(i)
        if i == '\n':
            singe_list.append(i)
            total_list.append(singe_list)
            singe_list = []
        else:
            singe_list.append(i)
# print(total_list)
print(total_list)
new_total_list = []
for i in total_list:
    for j in i:
        if j != '\n' and 'B-Location' in j:
            new_total_list.append(i)
print(new_total_list)
x_train, x_test = train_test_split(new_total_list, test_size=0.1)
print(len(x_train))
print(len(x_test))

with open('../data/valid_bmes.txt', 'w', encoding='utf-8') as f1:
    for i in x_test:
        for j in i:
            f1.write(j)

with open('../data/train_bmes.txt', 'w', encoding='utf-8') as f2:
    for i in x_train:
        for j in i:
            f2.write(j)
