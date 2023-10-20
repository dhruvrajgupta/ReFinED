import json
from refined.resource_management.lmdb_wrapper import LmdbImmutableDict
from typing import Mapping, List, Tuple, Dict, Any, Set


def main():
    idx_to_qcode = {}
    lmdb_file = "/home/fr/fr_fr/fr_dg262/.cache/refined/wikidata_data/qcode_to_idx.lmdb"
    class_de_file = "/home/fr/fr_fr/fr_dg262/ReFinED/src/refined/offline_data_generation/data/de/class_to_idx.json"

    qcode_to_idx: Mapping[str, int] = LmdbImmutableDict(lmdb_file)

    lmdb_txn = qcode_to_idx.env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        idx_to_qcode[int(value.decode("utf-8"))] = key.decode("utf-8")
    
    print("Writing to file...")
    with open("xyz.jsonl", 'w') as f:
        for key, value in sorted(idx_to_qcode.items()):
            f.write(json.dumps({'qcode': value, 'index': key}) + '\n')
    

if __name__ == "__main__":
    main()