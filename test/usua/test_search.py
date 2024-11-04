import requests
from bs4 import BeautifulSoup
import argparse
import json
from termcolor import colored

# https://github.com/bravekingzhang/search-engine-tool/issues/1


def get_page_content(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.get_text().replace("\n", "")
        return main_content[:300]
    except requests.RequestException as e:
        print(f"Error fetching page content: {e}")
        return "无法获取内容"


def scrape_bing_search_results(query, num_results=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    }
    url = f"https://bing.com/search?q={query}&qs=n&form=QBRE&sp=-1&lq=0"
    skip_keywords = ["zhihu", "zhidao.baidu", "wen.baidu"]

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    count = 0

    for result in soup.select(".b_algo"):
        if count >= num_results:
            break

        link_element = result.select_one("a")
        if not link_element:
            continue

        href = link_element["href"]
        if any(keyword in href for keyword in skip_keywords):
            continue
        title = link_element.text

        abstract_element = result.select_one(".b_caption p")
        abstract = abstract_element.get_text() if abstract_element else ""

        # 获取详细内容
        content = get_page_content(href, headers)

        results.append(
            {"href": href, "title": title, "abstract": abstract, "content": content}
        )
        count += 1

    # 如果结果数量不足 num_results，填充空结果
    while len(results) < num_results:
        print(f"{soup}")
        results.append({"href": "", "title": "", "abstract": "", "content": "问题敏感"})

    return results


def main():
    parser = argparse.ArgumentParser(description="Bing Search Results Scraper")
    parser.add_argument("-q", "--query", help="The search query", required=True)
    parser.add_argument(
        "-c", "--count", type=int, default=5, help="Number of results to retrieve"
    )
    args = parser.parse_args()

    search_results = scrape_bing_search_results(args.query, args.count)
    for result in search_results:
        print(colored(json.dumps(result, indent=2, ensure_ascii=False), "yellow"))


if __name__ == "__main__":
    # python test/usua/test_search.py -q 今天上海天气 -c 2
    # python test/usua/test_search.py -q 大道争锋 -c 2
    main()
