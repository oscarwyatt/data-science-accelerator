from bs4 import BeautifulSoup, SoupStrainer

def extract_content_item_body(content_item):
    try:
        body_soup = BeautifulSoup(content_item["details"]["body"], features='html.parser')
        return body_soup.get_text().replace('\n', '')
    except:
        return ""

def number_of_links(content_item):
    try:
        return len(BeautifulSoup(content_item["details"]["body"], features='html.parser', parse_only=SoupStrainer('a')))
    except:
        return 0

def number_of_words(content_item):
    return len(extract_content_item_body(content_item).split(" "))

def number_of_translations(content_item):
    return len(content_item.get("links", {}).get("available_translations", []))