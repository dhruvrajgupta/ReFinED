import os
from argparse import ArgumentParser
from refined.utilities.general_utils import get_logger
from refined.offline_data_generation.process_wiki import build_redirects
from refined.offline_data_generation.clean_wikipedia import preprocess_wikipedia
from refined.offline_data_generation.merge_files_and_extract_links import merge_files_and_extract_links
from refined.offline_data_generation.process_wikidata_dump import build_wikidata_lookups
from refined.resource_management.loaders import load_pem, load_labels, load_instance_of
from refined.offline_data_generation.preprocessing_utils import download_url_with_progress_bar
from refined.offline_data_generation.generate_pem import build_pem_lookup
from refined.offline_data_generation.generate_descriptions_tensor import create_description_tensor
from refined.offline_data_generation.class_selection import select_classes

from typing import Set, Dict, List
from refined.offline_data_generation.dataclasses_for_preprocessing import AdditionalEntity
from tqdm.auto import tqdm
import json

LANG = "de_"
OUTPUT_PATH = f'data'

LOG = get_logger(__name__)
keep_all_entities = True

os.makedirs(OUTPUT_PATH, exist_ok=True)

WIKIPEDIA_PAGE_IDS_FILE = LANG + 'wikipedia_page_ids.sql.gz'
WIKIPEDIA_REDIRECTS_FILE = LANG + 'wikipedia_redirects.sql.gz'
WIKIPEDIA_ARTICLES_FILE = LANG + 'wikipedia_articles.xml.bz2'
WIKIDATA_DUMP_FILE = 'wikidata.json.bz2'

AIDA_MEANS_URL = 'http://resources.mpi-inf.mpg.de/yago-naga/aida/download/aida_means.tsv.bz2'
AIDA_MEANS_FILE = 'aida_means.tsv.bz2'


def build_entity_index(pem_filename: str, output_path: str):
    pem = load_pem(pem_filename)
    all_qcodes: Set[str] = set()
    for qcode_probs in tqdm(pem.values(), total=18749702):
        all_qcodes.update(set(qcode_probs.keys()))
    qcode_to_index = {qcode: qcode_idx + 1 for qcode_idx, qcode in enumerate(list(all_qcodes))}
    # all_qcodes are randomly placed so index becomes random too....does this affect??

    with open(os.path.join(output_path, 'qcode_to_idx.json.part'), 'w') as fout:
        for k, v in qcode_to_index.items():
            fout.write(json.dumps({'qcode': k, 'index': v}) + '\n')
    os.rename(os.path.join(output_path, 'qcode_to_idx.json.part'), os.path.join(output_path, 'qcode_to_idx.json'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--debug', type=str,
                        default="n",
                        help="y or n", )
    parser.add_argument('--additional_entities_file', type=str)
    cli_args = parser.parse_args()
    debug = cli_args.debug.lower() == 'y'

    # Stripping of '_' from LANG
    lang = LANG[:-1]
    lang_dir = f'{OUTPUT_PATH}/{lang}'

    LOG.info('Step 2) Processing Wikidata dump to build lookups and sets.')
    args = {'dump_file_path': os.path.join(OUTPUT_PATH, WIKIDATA_DUMP_FILE),
            'output_dir': OUTPUT_PATH, 'overwrite_output_dir': True, 'test': debug}
    # Check and see if en wikidata is built
    LOG.info('Part 1. Building lookups and sets for Language - en')
    if not os.path.exists(os.path.join(f'{OUTPUT_PATH}/en')):
        build_wikidata_lookups(args_override=args, lang='en')

    LOG.info('Part 2. Building lookups and sets for Language - de')
    if not os.path.exists(os.path.join(f'{OUTPUT_PATH}/de/dewiki.json')):
        build_wikidata_lookups(args_override=args, lang='de')

    LOG.info('Step 3) Processing Wikipedia redirects dump.')
    args = {'page_sql_gz_filepath': os.path.join(lang_dir, WIKIPEDIA_PAGE_IDS_FILE),
            'redirect_sql_gz_filepath': os.path.join(lang_dir, WIKIPEDIA_REDIRECTS_FILE),
            'output_dir': lang_dir,
            'overwrite_output_dir': True,
            'test': debug}
    if not os.path.exists(os.path.join(lang_dir, 'redirects.json')):
        build_redirects(args=args, lang=lang)

    LOG.info('Step 4) Extract text from Wikipedia dump.')
    if not os.path.exists(os.path.join(lang_dir, 'wikipedia_links_aligned.json')):
        preprocess_wikipedia(dump_path=os.path.join(lang_dir, WIKIPEDIA_ARTICLES_FILE),
                             save_path=os.path.join(lang_dir, 'preprocessed_wikipedia'))
        merge_files_and_extract_links(input_dir=os.path.join(lang_dir, 'preprocessed_wikipedia'),
                                      resources_dir=OUTPUT_PATH, output_dir=OUTPUT_PATH, lang=lang)
        
    LOG.info('Step 5) Building PEM lookup.')
    # additional entity set file
    # {label: "label",
    # alias: ["alias_1", "alias2"],
    # entity_type: ["qcode_1", "qcode_2"],
    # entity_description: "english description"
    # }
    # A1, A2 instead of Q1, Q2
    additional_entities: List[AdditionalEntity] = []

    if cli_args.additional_entities_file is not None:
        print('Adding additional entities')
        with open(cli_args.additional_entities_file, 'r') as f:
            for line in tqdm(f, desc='Loading additional entities'):
                line = json.loads(line)
                additional_entities.append(AdditionalEntity(**line))
    
    # add extra human and fictional humans to human qcodes
    # TODO add fictional human to original human qcodes as well
    if additional_entities is not None and len(additional_entities) > 0:
        instance_of = load_instance_of(os.path.join(f'{OUTPUT_PATH}/common', 'instance_of_p31.json'), is_test=debug)
        human_qcodes: Set[str] = set()
        for qcode, classes in tqdm(instance_of.items(), desc='Adding human qcodes from instance_of'):
            if 'Q5' in classes or 'Q15632617' in classes:
                human_qcodes.add(qcode)

        for additional_entity in tqdm(additional_entities, desc='Adding human qcodes from additional entities'):
            if 'Q5' in additional_entity.entity_types or 'Q15632617' in additional_entity.entity_types:
                human_qcodes.add(additional_entity.entity_id)

        with open(os.path.join(f'{OUTPUT_PATH}/common', 'human_qcodes.json'), 'w') as f_out:
            f_out.write('\n'.join(human_qcodes))
        del instance_of

    if not os.path.exists(os.path.join(OUTPUT_PATH, AIDA_MEANS_FILE)):
        download_url_with_progress_bar(url=AIDA_MEANS_URL, output_path=os.path.join(OUTPUT_PATH, AIDA_MEANS_FILE))
    if not os.path.exists(os.path.join(f'{OUTPUT_PATH}/{lang}', 'wiki_pem.json')):
        build_pem_lookup(aligned_wiki_file=os.path.join(f'{OUTPUT_PATH}/{lang}', 'wikipedia_links_aligned.json'),
                         output_dir=OUTPUT_PATH, resources_dir=OUTPUT_PATH, keep_all_entities=keep_all_entities,
                         additional_entities=additional_entities,
                         is_test=debug, lang=lang)
        
    LOG.info('Step 6) Building entity index from PEM.')
    if not os.path.exists(os.path.join(lang_dir, 'qcode_to_idx.json')):
        build_entity_index(os.path.join(lang_dir, 'wiki_pem.json'), lang_dir)

    # build descriptions (include labels without descriptions, maybe some alts as well should keep it short)
    LOG.info('Step 7) Building descriptions tensor.')
    if not os.path.exists(os.path.join(lang_dir, 'descriptions_tns.pt')):
        create_description_tensor(output_path=lang_dir,
                                  qcode_to_idx_filename=os.path.join(lang_dir, 'qcode_to_idx.json'),
                                  desc_filename=os.path.join(lang_dir, 'desc.json'),
                                  label_filename=os.path.join(lang_dir, 'qcode_to_label.json'),
                                  wiki_to_qcode=os.path.join(lang_dir, f'{lang}wiki.json'),
                                  additional_entities=additional_entities,
                                  keep_all_entities=keep_all_entities,
                                  is_test=debug)
        
    LOG.info('Step 8) Selecting classes tensor.')
    if not os.path.exists(os.path.join(lang_dir, 'chosen_classes.txt')):
        select_classes(resources_dir=OUTPUT_PATH, is_test=debug, lang=lang)


if __name__ == "__main__":
    main()