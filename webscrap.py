import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
from playwright.sync_api import sync_playwright
import time

URL = "https://www.husqvarna.com/lt/grandininiai-pjuklai/"


def scrape_husqvarna():
    print("Playwright scraperis paleidžiamas...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=30)
        context = browser.new_context(
            locale="lt-LT",
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"Atidarome: {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(6)

        # Priimame slapukus
        for sel in ["#onetrust-accept-btn-handler", "button:has-text('Accept')", "button:has-text('Priimti')"]:
            try:
                page.locator(sel).click(timeout=4000)
                print("Slapukai priimti.")
                time.sleep(2)
                break
            except Exception:
                continue

        # Laukiame kol GTM dataLayer užsipildys
        print("Laukiame produktų duomenų...")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        time.sleep(3)

        # Ištraukiame dataLayer iš puslapio JavaScript konteksto
        dl_raw = page.evaluate("() => typeof dataLayer !== 'undefined' ? JSON.stringify(dataLayer) : null")

        browser.close()

    if not dl_raw:
        print("dataLayer nerasta - produktų surinkti nepavyko.")
        return pd.DataFrame()

    dl = json.loads(dl_raw)
    all_products = []

    for entry in dl:
        if not isinstance(entry, dict):
            continue
        ecommerce = entry.get("ecommerce", {})
        items = ecommerce.get("items", [])
        if not items:
            continue
        print(f"Rastas GTM event '{entry.get('event')}' su {len(items)} produktais.")
        for item in items:
            name = item.get("item_name") or item.get("name") or ""
            price = item.get("price") or item.get("value") or ""
            category = item.get("item_category") or ""
            brand = item.get("item_brand") or ""
            if name:
                all_products.append({
                    "Pavadinimas": name,
                    "Kategorija": category,
                    "Kaina (EUR)": price,
                    "Prekės ženklas": brand,
                })

    if not all_products:
        print("GTM dataLayer nerasta produktų su ecommerce duomenimis.")
        return pd.DataFrame()

    df = pd.DataFrame(all_products).drop_duplicates(subset=["Pavadinimas"])
    return df


df = scrape_husqvarna()

print("\n--- Surinkti duomenys ---")
if not df.empty:
    print(df.to_string(index=False))
    df.to_csv("husqvarna_pjuklai.csv", index=False, encoding="utf-8-sig")
    print(f"\nIšsaugota {len(df)} įrašų į husqvarna_pjuklai.csv")
else:
    print("Duomenų surinkti nepavyko.")
