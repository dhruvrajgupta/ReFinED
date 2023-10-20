import json


def main():
    class_en_file = "/home/fr/fr_fr/fr_dg262/.cache/refined/wikipedia_data/class_to_idx.json"
    with open(class_en_file) as f:
        class_en_data = json.load(f)
    
    class_de_file = "/home/fr/fr_fr/fr_dg262/ReFinED/src/refined/offline_data_generation/data/de/class_to_idx.json"
    with open(class_de_file) as f:
        class_de_data = json.load(f)

    de_keys = []
    en_keys = []

    for key, value in class_de_data.items():
        if key in class_en_data.keys():
            de_keys.append(key)
            # print(value)
            # print(class_en_data[key])
            # print("")
    print(de_keys)
    print(len(de_keys))

    # for key, value in class_en_data.items():
    #     if key not in class_de_data.keys():
    #         en_keys.append(key)
    #         # print(value)
    #         # print(class_en_data[key])
    #         # print("")
    # print(en_keys)
    # print(len(en_keys))


if __name__ == "__main__":
    main()