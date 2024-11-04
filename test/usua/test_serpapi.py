from serpapi import GoogleSearch

# pip install google-search-results
params = {
    "engine": "bing",
    "q": "大道争锋",
    "cc": "TW",
    "api_key": "138f7017c1a5611a353fa66c73417498272f3568d7a0629747ef3a4bdc500433",
}

search = GoogleSearch(params)
results = search.get_dict()
print(f"{results}")
# python test/usua/test_serpapi.py
