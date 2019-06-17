from collections import Counter

def _read_labels():

    with open("labels.json", "r") as f:
        type_dict = eval(f.read().strip())

    return type_dict

def main():

    with open("test.txt","r") as f:

        data_list = f.readlines()
        labels_list = []
        for i in data_list:

            data = i.split(":")[0]
            labels_list.append(data)
            # print(data)
            # break
        # print(data_list[:5])
    print(Counter(labels_list))
    labels_data = Counter(labels_list)
    sum = 0
    for i in labels_data.values():
        sum += 1
    print("sum", sum)
    data_set = set(labels_list)
    print(Counter(data_set))
    print(len(data_set))
    labels_list = _read_labels()
    for label in data_set:

        if label not in labels_list:

            print(label)


if __name__ == '__main__':

    main()
    # print(_read_labels())
