import bz2
import json

from tqdm.auto import tqdm
import logging
import argparse
import os
from types import SimpleNamespace

# TODO: check for duplicates if processed on different langs
def extract_useful_info(entity, lang):
    qcode = entity['id']
    if lang in entity['labels']:
        entity_label = entity['labels'][lang]['value']
    else:
        entity_label = None
    if lang in entity['descriptions']:
        entity_desc = entity['descriptions'][lang]['value']
    else:
        entity_desc = None
    if lang in entity['aliases']:
        entity_aliases = [alias['value'] for alias in entity['aliases'][lang]]
    else:
        entity_aliases = []
    if 'sitelinks' in entity:
        sitelinks = entity['sitelinks']
    else:
        sitelinks = {}
    if f'{lang}wiki' in sitelinks:
        langwiki_title = sitelinks[f'{lang}wiki']['title']
    else:
        langwiki_title = None

    sitelinks_cnt = len(sitelinks.items())
    statements_cnt = 0
    triples = {}
    for pcode, objs in entity['claims'].items():
        # group by pcode -> [list of qcodes]
        for obj in objs:
            statements_cnt += 1
            if not obj['mainsnak']['datatype'] == 'wikibase-item' or obj['mainsnak']['snaktype'] == 'somevalue' \
                    or 'datavalue' not in obj['mainsnak']:
                continue
            if pcode not in triples:
                triples[pcode] = []
            triples[pcode].append(obj['mainsnak']['datavalue']['value']['id'])
    return {'qcode': qcode, 'label': entity_label, 'desc': entity_desc,
            'aliases': entity_aliases, 'sitelinks_cnt': sitelinks_cnt, f'{lang}wiki_title': langwiki_title,
            'statements_cnt': statements_cnt, 'triples': triples}


def build_wikidata_lookups(args_override=None, lang=None):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{lang}_build_wikidata_lookup_log.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger.info("This is an INFO test message using Python logging")

    if lang is None:
        print("ERROR: Language not specified")
        return
    if args_override is None:
        parser = argparse.ArgumentParser(description='Build lookup dictionaries from Wikidata JSON dump.')
        parser.add_argument(
            "--dump_file_path",
            default='latest-all.json.bz2',
            type=str,
            help="file path to JSON Wikidata dump file (latest-all.json.bz2)"
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default='output',
            help="Directory where the lookups will be stored"
        )
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="Overwrite the content of the output directory"
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="mode for testing (only processes first 500 lines)"
        )
        args = parser.parse_args()
    else:
        args = SimpleNamespace(**args_override)
    args.output_dir = args.output_dir.rstrip('/')
    number_lines_to_process = 500 if args.test else 1e20
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create lang specific directory
    lang_dir = f'{args.output_dir}/{lang}'
    if not os.path.exists(lang_dir):
        os.makedirs(lang_dir)

    # Create common directory
    common_dir = f'{args.output_dir}/common'
    if not os.path.exists(common_dir):
        os.makedirs(common_dir)

    if os.path.exists(f'{lang_dir}/qcode_to_label.json'):
        # the script has already been run so do not repeat the work.
        return

    i = 0

    # duplicate names in output_files to make removing .part easier after pre-processing has finished.
    filenames = [
        f'{common_dir}/sitelinks_cnt.json.part',
        f'{common_dir}/statements_cnt.json.part',
        f'{lang_dir}/{lang}wiki.json.part',
        f'{lang_dir}/desc.json.part',
        f'{lang_dir}/aliases.json.part',
        f'{lang_dir}/qcode_to_label.json.part',
        f'{common_dir}/instance_of_p31.json.part',
        f'{common_dir}/country_p17.json.part',
        f'{common_dir}/sport_p641.json.part',
        f'{common_dir}/occupation_p106.json.part',
        f'{common_dir}/subclass_p279.json.part',
        f'{lang_dir}/pcodes.json.part',
        f'{common_dir}/human_qcodes.json.part',
        f'{common_dir}/disambiguation_qcodes.txt.part',
        f'{common_dir}/triples.json.part',
        f'{common_dir}/located_in_p131.json.part',
    ]

    output_files = {
        'sitelinks_cnt': open(f'{common_dir}/sitelinks_cnt.json.part', 'w'),
        'statements_cnt': open(f'{common_dir}/statements_cnt.json.part', 'w'),
        f'{lang}wiki': open(f'{lang_dir}/{lang}wiki.json.part', 'w'),
        'desc': open(f'{lang_dir}/desc.json.part', 'w'),
        'aliases': open(f'{lang_dir}/aliases.json.part', 'w'),
        'label': open(f'{lang_dir}/qcode_to_label.json.part', 'w'),
        'instance_of_p31': open(f'{common_dir}/instance_of_p31.json.part', 'w'),
        'country_p17': open(f'{common_dir}/country_p17.json.part', 'w'),
        'sport_p641': open(f'{common_dir}/sport_p641.json.part', 'w'),
        'occupation_p106': open(f'{common_dir}/occupation_p106.json.part', 'w'),
        'subclass_p279': open(f'{common_dir}/subclass_p279.json.part', 'w'),
        'properties': open(f'{lang_dir}/pcodes.json.part', 'w'),
        'humans': open(f'{common_dir}/human_qcodes.json.part', 'w'),
        'disambiguation': open(f'{common_dir}/disambiguation_qcodes.txt.part', 'w'),
        'triples': open(f'{common_dir}/triples.json.part', 'w'),
        'located_in_p131': open(f'{common_dir}/located_in_p131.json.part', 'w'),
    }

    with bz2.open(args.dump_file_path, 'rb') as f:
        for line in tqdm(f, total=80e6):
            i += 1
            if i%1e6 == 0:
                logger.info(f"{i} out of {80e6}")
            if len(line) < 3:
                continue
            line = line.decode('utf-8').rstrip(',\n')
            line = json.loads(line)
            entity_content = extract_useful_info(line, lang)
            qcode = entity_content['qcode']
            if 'P' in qcode:
                output_files['properties'].write(json.dumps({'qcode': qcode, 'values': entity_content}) + '\n')

            if entity_content['sitelinks_cnt']:
                output_files['sitelinks_cnt'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['sitelinks_cnt']}) + '\n')

            if entity_content['statements_cnt']:
                output_files['statements_cnt'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['statements_cnt']}) + '\n')

            if entity_content[f'{lang}wiki_title']:
                output_files[f'{lang}wiki'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content[f'{lang}wiki_title']}) + '\n')

            if entity_content['desc']:
                output_files['desc'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['desc']}) + '\n')

            if entity_content['aliases']:
                output_files['aliases'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['aliases']}) + '\n')

            if entity_content['label']:
                output_files['label'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['label']}) + '\n')

            # <instance of, class>
            if 'P31' in entity_content['triples']:
                output_files['instance_of_p31'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P31']}) + '\n')
                if 'Q5' in entity_content['triples']['P31'] or 'Q15632617' in entity_content['triples']['P31']:
                    output_files['humans'].write(str(qcode) + '\n')
                if 'Q4167410' in entity_content['triples']['P31'] or 'Q22808320' in entity_content['triples']['P31']:
                    output_files['disambiguation'].write(str(qcode) + '\n')

            if 'P131' in entity_content['triples']:
                output_files['located_in_p131'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P131']}) + '\n')

            # <country, class>
            if 'P17' in entity_content['triples']:
                output_files['country_p17'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P17']}) + '\n')

            # <sport, class>
            if 'P641' in entity_content['triples']:
                output_files['sport_p641'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P641']}) + '\n')

            # <occupation, class>
            if 'P106' in entity_content['triples']:
                output_files['occupation_p106'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P106']}) + '\n')

            # <subclass, class>
            if 'P279' in entity_content['triples']:
                output_files['subclass_p279'] \
                    .write(json.dumps({'qcode': qcode, 'values': entity_content['triples']['P279']}) + '\n')

            output_files['triples'].write(json.dumps({'qcode': qcode, 'triples': entity_content['triples']}) + '\n')

            if i > number_lines_to_process:
                break

    for file in output_files.values():
        file.close()

    if not args.test:
        for filename in filenames:
            os.rename(filename, filename.replace('.part', ''))


if __name__ == '__main__':
    build_wikidata_lookups()
