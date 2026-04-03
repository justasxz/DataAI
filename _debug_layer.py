import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=30)
    context = browser.new_context(locale="lt-LT", viewport={"width": 1366, "height": 768})
    page = context.new_page()
    page.goto("https://www.husqvarna.com/lt/grandininiai-pjuklai/", wait_until="domcontentloaded")
    time.sleep(6)

    try:
        page.locator("#onetrust-accept-btn-handler").click(timeout=4000)
        time.sleep(3)
    except Exception:
        pass

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    time.sleep(3)

    # trackingLayer
    layer = page.evaluate("() => typeof trackingLayer !== 'undefined' ? JSON.stringify(trackingLayer) : null")
    if layer:
        data = json.loads(layer)
        print("=== trackingLayer ===")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])
    else:
        print("trackingLayer nerasta")

    # dataLayer (GTM - dažnai turi product impressions)
    dl = page.evaluate("() => typeof dataLayer !== 'undefined' ? JSON.stringify(dataLayer) : null")
    if dl:
        ddata = json.loads(dl)
        print("\n=== dataLayer ===")
        for entry in ddata:
            txt = json.dumps(entry, ensure_ascii=False)
            if any(k in txt for k in ["product", "name", "impression", "ecommerce"]):
                print(json.dumps(entry, ensure_ascii=False, indent=2)[:2000])
                print("---")

    # Ieskome produktų list puslapyje
    html_snippet = page.evaluate("""() => {
        const el = document.querySelector('[class*=product-list__grid], [class*=hbd-product-list__grid], [class*=product-list__items]');
        return el ? el.innerHTML.slice(0, 3000) : 'NOT FOUND';
    }""")
    print("\n=== Product list grid HTML ===")
    print(html_snippet[:3000])

    browser.close()
