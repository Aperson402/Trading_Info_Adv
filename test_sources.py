"""
Run this on your Mac to find which feed URLs actually work:
  python test_sources.py

Paste the output back and I'll update sources.py accordingly.
"""
import urllib.request
import sys

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

CANDIDATES = [
    # --- EIA ---
    ("EIA", "https://www.eia.gov/rss/press_releases.xml"),
    ("EIA", "https://www.eia.gov/petroleum/pressreleases/rss/feed.xml"),
    ("EIA", "https://www.eia.gov/totalenergy/data/monthly/rss/feed.xml"),

    # --- Baker Hughes ---
    ("Baker Hughes", "https://rigcount.bakerhughes.com/rss"),
    ("Baker Hughes", "https://rigcount.bakerhughes.com/sitemap.xml"),  # check site structure

    # --- AP News ---
    ("AP", "https://apnews.com/rss/apf-business"),
    ("AP", "https://apnews.com/rss/apf-topnews"),
    ("AP", "https://apnews.com/hub/business?format=rss"),
    ("AP", "https://apnews.com/hub/energy?format=rss"),

    # --- Reuters ---
    ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/USenergyNews"),
    ("Reuters", "https://www.reuters.com/tools/rss"),  # check if RSS page loads

    # --- State Dept ---
    ("State", "https://www.state.gov/press-releases/"),
    ("State", "https://www.state.gov/feed/"),
    ("State", "https://www.state.gov/rss-feeds/press-releases/"),

    # --- FT ---
    ("FT", "https://www.ft.com/rss/home/uk"),
    ("FT", "https://www.ft.com/rss/home"),
    # --- OPEC RSS (new) ---
    ("OPEC", "https://www.opec.org/opec_web/en/pressreleases/rss"),
    # --- State Dept Near East RSS (new) ---
    ("State Near East", "https://www.state.gov/rss-feeds/Near-East/"),
    ("State full feed",  "https://www.state.gov/feed/"),
    # --- IEA (replacing Baker Hughes) ---
    ("IEA", "https://www.iea.org/feed"),
    ("IEA news", "https://www.iea.org/news.rss"),
]

print(f"Testing {len(CANDIDATES)} URLs...\n")
for source, url in CANDIDATES:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read(800).decode("utf-8", errors="ignore")
        is_rss = any(tag in body for tag in ["<rss", "<feed", "<channel", "<?xml"])
        tag = "RSS✅" if is_rss else "HTML✅"
        print(f"  {tag}  [{source}]  {url}")
    except urllib.error.HTTPError as e:
        print(f"  {e.code}❌  [{source}]  {url}")
    except Exception as e:
        print(f"  ERR❌  [{source}]  {url}  ({type(e).__name__}: {str(e)[:50]})")

print("\nDone. Paste this output back to update sources.py.")