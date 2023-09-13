import json
import time

import requests
import wptools


def main():
    lang = "de"
    updated_dicts = []

    with open(f'data/{lang}_wikipedia_links_aligned_10_unformatted.jsonl', 'r') as json_file:
        total_start_time = time.time()
        print("Stats:")
        print("*" * 50)
        for i, line in enumerate(json_file):
            # Just for 2 articles
            if i == 2:
                break
            line = json.loads(line)
            id = line["id"]
            revid = line["revid"]
            url = line["url"]
            title = line["title"]
            text = line["text"]
            categories = line["categories"]
            hyperlinks = line["hyperlinks"]
            hyperlinks_clean = line["hyperlinks_clean"]

            error_counts = 0
            updated_hyperlinks = []
            doc_start_time = time.time()

            print(f"Title: {title}")
            print(f"Article: {url}")
            for hyperlink in hyperlinks:
                uri = hyperlink["uri"]
                qcode, error = get_qcode(uri, hyperlink["surface_form"], lang)
                if error:
                    error_counts += 1
                if qcode is not None:
                    hyperlink["qcode"] = qcode
                    updated_hyperlinks.append(hyperlink)

            print(f"Error Counts: {error_counts}")
            line["hyperlinks_clean"] = updated_hyperlinks
            updated_dicts.append(line)
            doc_end_time = time.time()
            print(f"Time taken to process document: {doc_end_time - doc_start_time}s\n")
            print("*" * 50)

        with open('data/de_wikipedia_links_aligned_10_with_qcodes.jsonl', 'w') as file:
            for line in updated_dicts:
                file.write(json.dumps(line) + "\n")
        total_end_time = time.time()
        print(f"Total time taken: {total_end_time - total_start_time}s")


def get_qcode(uri, surface_form, lang):
    try:
        page = wptools.page(uri, lang="de", silent=True).get()
        qcode = page.data["wikibase"]
        return qcode, False
    except Exception as e:
        api_response = get_response_from_api(uri, lang)
        qcode = get_qcode_from_api(api_response)
        if qcode is None:
            # Try with english
            api_response = get_response_from_api(uri, "en")
            qcode = get_qcode_from_api(api_response)
        else:
            return qcode, False

        if qcode is None:
            print("=" * 20)
            print(f"uri - {uri}")
            print(f"surface_form - {surface_form}")
            print(
                f"error - https://{lang}.wikipedia.org/w/api.php?action=query&format=json&titles=&prop=pageprops"
                f"&titles={uri}")
            print("=" * 20)
            return None, True
        else:
            return qcode, False


def get_response_from_api(uri, lang):
    return requests.get(
        f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&titles={uri}") \
        .json()


def get_qcode_from_api(api_response):
    for page_id, e_page in api_response["query"]["pages"].items():

        if page_id == "-1":
            return None

        if "pageprops" in e_page.keys() and "wikibase_item" in "wikibase_item" in e_page["pageprops"].keys():
            qcode = e_page["pageprops"]["wikibase_item"]
            return qcode


if __name__ == "__main__":
    main()
