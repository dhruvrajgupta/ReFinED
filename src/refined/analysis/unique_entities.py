lang = "pt"

dir_path = f"../offline_data_generation/data/organised_data_dir_{lang}"

import json

total_qcodes_list = []

# for datasplits 10 to 100
for i in range(10, 101, 10):

    qcodes_list = []

    with open(f"{dir_path}/datasets/{lang}_wikipedia_links_aligned_spans_train_{i}p.json") as f:
        for line in f:
            data = json.loads(line)
            hyperlinks = data["hyperlinks_clean"]
            for hyperlink in hyperlinks:
                qcodes_list.append(hyperlink["qcode"])
                total_qcodes_list.append(hyperlink["qcode"])
            # qcodes_list.append()
    print(f"{i}p : ")
    # print(len(qcodes_list))
    print(len(set(qcodes_list)))
    qcodes_list.clear()
    print()

with open(f"{dir_path}/datasets/{lang}_wikipedia_links_aligned_spans_val_1e4_conf.json") as f:
    for line in f:
        data = json.loads(line)
        hyperlinks = data["hyperlinks_clean"]
        for hyperlink in hyperlinks:
            qcodes_list.append(hyperlink["qcode"])
            total_qcodes_list.append(hyperlink["qcode"])
        # qcodes_list.append()
print(f"{i}p : ")
# print(len(qcodes_list))
print(len(set(qcodes_list)))
qcodes_list.clear()
print()


print("\n\n")
print(len(total_qcodes_list))
print(len(set(total_qcodes_list)))