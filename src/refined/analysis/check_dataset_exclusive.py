import json


def main():
    test_de_ids = []
    test_de_file = "/home/fr/fr_fr/fr_dg262/ReFinED/src/refined/offline_data_generation/data/de/dataset/de_wikipedia_links_aligned_test_1e4.json"
    with open(test_de_file) as f:
        test_de_data = [json.loads(line) for line in f]
        for line in test_de_data:
            id = int(line[u'id'])
            test_de_ids.append(id)

    eval_de_ids = []
    eval_de_file = "/home/fr/fr_fr/fr_dg262/ReFinED/src/refined/offline_data_generation/data/de/dataset/de_wikipedia_links_aligned_eval_1e4.json"
    with open(eval_de_file) as f:
        eval_de_data = [json.loads(line) for line in f]
        for line in eval_de_data:
            id = int(line[u'id'])
            eval_de_ids.append(id)
    
    train_de_ids = []
    train_de_file = "/home/fr/fr_fr/fr_dg262/ReFinED/src/refined/offline_data_generation/data/de/dataset/de_wikipedia_links_aligned_train_1e5.json"
    with open(train_de_file) as f:
        train_de_data = [json.loads(line) for line in f]
        for line in train_de_data:
            id = int(line[u'id'])
            train_de_ids.append(id)

    
    print(len(test_de_ids))
    print(test_de_ids)
    print(len(eval_de_ids))
    print(len(train_de_ids))

    test = set(test_de_ids)
    train = set(train_de_ids)
    eval = set(eval_de_ids)

    print(test & train)
    print(test & eval)

    print(train & test)
    print(train & eval)

    print(eval & test)
    print(eval & train)

if __name__ == "__main__":
    main()