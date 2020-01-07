import json


def run():
    data_path = "data/corpus/train.json"
    with open(data_path, encoding='utf-8') as fin:
        data_set = []
        j = 0
        for lidx, line in enumerate(fin):
            j += 1
            print(111)
            sample = json.loads(line.strip())
            data_set.append(sample)
            # if j == 100000:
            #     break
            print(1)
        with open("train.json", 'w', encoding='utf-8') as json_file:
            json.dump(sample, json_file, ensure_ascii=False)


if __name__ == '__main__':
    run()
