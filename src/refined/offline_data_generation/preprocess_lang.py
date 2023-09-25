import os
from argparse import ArgumentParser
from refined.utilities.general_utils import get_logger
from refined.offline_data_generation.process_wiki import build_redirects
from refined.offline_data_generation.clean_wikipedia import preprocess_wikipedia
from refined.offline_data_generation.merge_files_and_extract_links import merge_files_and_extract_links
from refined.offline_data_generation.process_wikidata_dump import build_wikidata_lookups

LANG = "de_"
OUTPUT_PATH = f'data'

LOG = get_logger(__name__)

os.makedirs(OUTPUT_PATH, exist_ok=True)

WIKIPEDIA_PAGE_IDS_FILE = LANG + 'wikipedia_page_ids.sql.gz'
WIKIPEDIA_REDIRECTS_FILE = LANG + 'wikipedia_redirects.sql.gz'
WIKIPEDIA_ARTICLES_FILE = LANG + 'wikipedia_articles.xml.bz2'
WIKIDATA_DUMP_FILE = 'wikidata.json.bz2'

def main():
    parser = ArgumentParser()
    parser.add_argument('--debug', type=str,
                        default="n",
                        help="y or n", )
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
        return

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
        preprocess_wikipedia(dump_path=os.path.join(OUTPUT_PATH, WIKIPEDIA_ARTICLES_FILE),
                             save_path=os.path.join(lang_dir, 'preprocessed_wikipedia'))
        merge_files_and_extract_links(input_dir=os.path.join(lang_dir, 'preprocessed_wikipedia'),
                                      resources_dir=OUTPUT_PATH, output_dir=OUTPUT_PATH, lang=lang)

if __name__ == "__main__":
    main()