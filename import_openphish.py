#!/usr/bin/env python3
"""Optional: download a small sample of REAL phishing URLs from the OpenPhish community feed.

NOT TESTED in the authoring environment (no network there). OpenPhish Community feed is free for
NON-COMMERCIAL use (university project is fine) - read their terms: https://openphish.com/
The feed contains URLs only (no message text), so it is useful for URL-feature experiments, not for Naive Bayes.
Never open these URLs in a browser.

Usage: python3 tools/import_openphish.py 50
"""
import csv, os, sys, urllib.request

FEED = "https://openphish.com/feed.txt"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
with urllib.request.urlopen(FEED, timeout=30) as r:
    urls = [l.strip() for l in r.read().decode("utf-8", "ignore").splitlines() if l.strip()][:n]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "openphish_urls_sample.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["url", "label", "source"])
    for u in urls:
        w.writerow([u, "PHISHING", FEED])
print(f"saved {len(urls)} URLs to {out}")
