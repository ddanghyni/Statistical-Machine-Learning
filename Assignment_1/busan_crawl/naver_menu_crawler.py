"""
네이버 지도 메뉴 가격 크롤러
==============================
Google Places API에서 수집한 가게들의 메뉴 가격을
네이버 지도에서 보완 수집

전략:
  1) Google 데이터의 가게명으로 네이버 지도 검색
  2) 검색 결과에서 네이버 place_id 추출 (entryIframe src)
  3) 메뉴 페이지(pcmap.place.naver.com/place/{id}/menu/list)에 직접 접근
  4) 메뉴명 + 가격 수집
  5) 가게별 메뉴 가격 중앙값 계산
  6) Google 데이터와 병합
"""

import asyncio
import csv
import json
import os
import re
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from playwright.async_api import async_playwright, Page, BrowserContext
import pandas as pd
import numpy as np

TIMEOUT = 20000
NAV_TIMEOUT = 30000
REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
OUTPUT_DIR = "output"


class NaverMenuCrawler:
    """네이버 지도에서 메뉴 가격만 수집"""

    def __init__(self, headless=True):
        self.headless = headless
        self.crawl_date = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _get_launch_options(self):
        return {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-gpu",
                "--window-size=1920,1080",
            ],
        }

    def _get_context_options(self):
        return {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "locale": "ko-KR",
            "timezone_id": "Asia/Seoul",
        }

    async def _random_delay(self):
        await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

    # ──────────────────────────────────────────
    # Step 1: 가게명으로 네이버 검색 → place_id 추출
    # ──────────────────────────────────────────
    async def _search_naver(self, page: Page, query: str) -> Optional[str]:
        """
        네이버 지도에서 검색 → place_id 추출
        단일 결과 → URL/entryIframe에서 바로 추출
        다중 결과 → searchIframe에서 첫 번째 클릭 → entryIframe에서 추출
        """
        try:
            await page.goto(
                "https://map.naver.com/", wait_until="domcontentloaded", timeout=NAV_TIMEOUT
            )
            await asyncio.sleep(2)

            # 검색창 찾기
            search_input = None
            for sel in ["input.input_search", "input#search-input", "input[type='text']"]:
                try:
                    search_input = await page.wait_for_selector(sel, state="visible", timeout=8000)
                    if search_input:
                        break
                except Exception:
                    continue

            if not search_input:
                return None

            await search_input.click()
            await asyncio.sleep(0.3)
            await search_input.fill(query)
            await asyncio.sleep(0.3)
            await search_input.press("Enter")
            await asyncio.sleep(3)

            # ── 방법 1: URL에서 place_id (단일 결과 → 바로 상세 페이지) ──
            match = re.search(r"/place/(\d+)", page.url)
            if match:
                return match.group(1)

            # ── 방법 2: entryIframe에서 추출 (단일 결과) ──
            try:
                entry_el = await page.wait_for_selector(
                    "iframe#entryIframe", state="visible", timeout=3000
                )
                if entry_el:
                    src = await entry_el.get_attribute("src") or ""
                    match = re.search(r"/place/(\d+)", src)
                    if match:
                        return match.group(1)
            except Exception:
                pass

            # ── 방법 3: searchIframe (다중 결과) → 첫 번째 클릭 ──
            try:
                await page.wait_for_selector("iframe#searchIframe", state="visible", timeout=5000)
                search_frame_el = await page.query_selector("iframe#searchIframe")
                if not search_frame_el:
                    return None

                search_frame = await search_frame_el.content_frame()
                if not search_frame:
                    return None

                # 첫 번째 결과 클릭
                clicked = await search_frame.evaluate("""
                    () => {
                        const item = document.querySelector('li.UEzoS');
                        if (!item) return false;
                        const link = item.querySelector('.place_bluelink') ||
                                     item.querySelector('a.tzwk0') ||
                                     item.querySelector('a');
                        if (link) { link.click(); return true; }
                        return false;
                    }
                """)

                if not clicked:
                    return None

                await asyncio.sleep(3)

                # entryIframe에서 place_id 추출
                try:
                    entry_el = await page.wait_for_selector(
                        "iframe#entryIframe", state="visible", timeout=8000
                    )
                    if entry_el:
                        src = await entry_el.get_attribute("src") or ""
                        match = re.search(r"/place/(\d+)", src)
                        if match:
                            return match.group(1)
                except Exception:
                    pass

                # URL fallback
                match = re.search(r"/place/(\d+)", page.url)
                if match:
                    return match.group(1)

            except Exception:
                pass

            return None

        except Exception:
            return None

    async def _search_and_get_naver_id(
        self, page: Page, store_name: str, district: str
    ) -> Optional[str]:
        """
        여러 검색 전략으로 네이버 place_id 찾기
        
        시도 순서:
          1) "가게명" 만으로 검색 (가장 정확)
          2) "가게명 부산" 으로 검색
          3) "가게명 행정구" 로 검색
        """
        # 시도 1: 가게명만
        naver_id = await self._search_naver(page, store_name)
        if naver_id:
            return naver_id

        # 시도 2: 가게명 + 부산
        naver_id = await self._search_naver(page, f"{store_name} 부산")
        if naver_id:
            return naver_id

        # 시도 3: 가게명 + 행정구
        if district:
            naver_id = await self._search_naver(page, f"{store_name} {district}")
            if naver_id:
                return naver_id

        return None

    # ──────────────────────────────────────────
    # Step 2: 메뉴 페이지에서 가격 추출 (새 탭)
    # ──────────────────────────────────────────
    async def _extract_menus(
        self, naver_place_id: str, context: BrowserContext
    ) -> List[Dict]:
        """메뉴 페이지 URL로 직접 접근 → 메뉴명 + 가격 추출"""
        menus = []
        menu_page = await context.new_page()

        try:
            menu_url = f"https://pcmap.place.naver.com/place/{naver_place_id}/menu/list"
            await menu_page.goto(menu_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await asyncio.sleep(2)

            raw = await menu_page.evaluate("""
                () => {
                    const results = [];

                    // 여러 셀렉터 시도
                    const selectors = [
                        'ul.czfPJ li',
                        'ul.E2jtL li',
                        '.place_section_content li',
                        '.order_list_area li',
                        'div.item_list li',
                    ];

                    let items = [];
                    for (const s of selectors) {
                        items = document.querySelectorAll(s);
                        if (items.length > 0) break;
                    }

                    for (const item of items) {
                        let name = '';
                        let price = null;

                        // 메뉴명
                        for (const s of ['span.lPzHi', 'span.place_bluelink', '.tit', '.name', 'span.name_menu']) {
                            const el = item.querySelector(s);
                            if (el) { name = el.textContent.trim(); if (name) break; }
                        }

                        // 가격
                        for (const s of ['div.GXS1X', '.price', 'em.price', 'span.price', '.price_menu']) {
                            const el = item.querySelector(s);
                            if (el) {
                                const text = el.textContent.trim();
                                const m = text.match(/[\\d,]+/);
                                if (m) {
                                    const p = parseInt(m[0].replace(/,/g, ''));
                                    if (p > 0) { price = p; break; }
                                }
                            }
                        }

                        if (name && price) {
                            results.push({name: name, price: price});
                        }
                    }

                    return results;
                }
            """)
            menus = raw or []

        except Exception:
            pass
        finally:
            await menu_page.close()

        return menus

    # ──────────────────────────────────────────
    # 메인: Google 데이터 기반 메뉴 가격 수집
    # ──────────────────────────────────────────
    async def crawl_menu_prices(self, google_csv_path: str, max_items: int = None):
        """
        Google Places 데이터의 각 가게에 대해 네이버에서 메뉴 가격 수집

        Args:
            google_csv_path: Google Places CSV 파일 경로
            max_items: 최대 처리 수 (테스트용, None이면 전부)
        """
        # Google 데이터 로드
        df = pd.read_csv(google_csv_path)

        # rating 있는 것만 (전처리 후 사용할 데이터)
        df = df.dropna(subset=["rating"])

        # 유효 행정구만
        valid_districts = [
            "해운대구", "부산진구", "금정구", "남구", "수영구",
            "중구", "동래구", "사하구", "북구", "사상구",
            "연제구", "영도구", "강서구", "동구", "서구", "기장군"
        ]
        df = df[df["district"].isin(valid_districts)]

        if max_items:
            df = df.head(max_items)

        print("=" * 60)
        print("네이버 지도 메뉴 가격 크롤러")
        print("=" * 60)
        print(f"대상 가게 수: {len(df)}개")

        # 결과 저장
        all_menus = []  # 메뉴 상세
        store_prices = []  # 가게별 가격 요약

        async with async_playwright() as p:
            browser = await p.chromium.launch(**self._get_launch_options())
            context = await browser.new_context(**self._get_context_options())
            page = await context.new_page()
            page.set_default_timeout(NAV_TIMEOUT)

            # 이미지 차단 (속도 향상)
            await page.route("**/*.{png,jpg,jpeg,gif,svg,webp}", lambda r: r.abort())

            success = 0
            fail = 0

            for idx, (_, row) in enumerate(df.iterrows(), 1):
                google_id = row["place_id"]
                name = row["name"]
                district = row.get("district", "")

                print(f"  [{idx}/{len(df)}] {name} ({district}) ", end="", flush=True)

                # Step 1: 네이버 place_id 검색
                naver_id = await self._search_and_get_naver_id(page, name, district)

                if not naver_id:
                    print("→ 네이버 검색 실패")
                    fail += 1
                    store_prices.append({
                        "google_place_id": google_id,
                        "name": name,
                        "naver_place_id": None,
                        "menu_count": 0,
                        "price_median": None,
                        "price_mean": None,
                        "price_min": None,
                        "price_max": None,
                        "status": "naver_not_found",
                    })
                    await self._random_delay()
                    continue

                # Step 2: 메뉴 가격 추출
                menus = await self._extract_menus(naver_id, context)

                if menus:
                    prices = [m["price"] for m in menus if m["price"] and m["price"] > 0]
                    price_median = int(np.median(prices)) if prices else None
                    price_mean = int(np.mean(prices)) if prices else None
                    price_min = min(prices) if prices else None
                    price_max = max(prices) if prices else None

                    print(f"→ ✓ 네이버ID={naver_id} | 메뉴 {len(menus)}개 | 중앙값 {price_median}원")
                    success += 1

                    # 메뉴 상세 저장
                    for m in menus:
                        all_menus.append({
                            "google_place_id": google_id,
                            "name": name,
                            "naver_place_id": naver_id,
                            "menu_name": m["name"],
                            "price": m["price"],
                        })

                    store_prices.append({
                        "google_place_id": google_id,
                        "name": name,
                        "naver_place_id": naver_id,
                        "menu_count": len(menus),
                        "price_median": price_median,
                        "price_mean": price_mean,
                        "price_min": price_min,
                        "price_max": price_max,
                        "status": "success",
                    })
                else:
                    print(f"→ △ 네이버ID={naver_id} | 메뉴 없음")
                    fail += 1
                    store_prices.append({
                        "google_place_id": google_id,
                        "name": name,
                        "naver_place_id": naver_id,
                        "menu_count": 0,
                        "price_median": None,
                        "price_mean": None,
                        "price_min": None,
                        "price_max": None,
                        "status": "no_menu",
                    })

                # 중간 저장 (50개마다)
                if idx % 50 == 0:
                    self._save_intermediate(store_prices, all_menus, idx)

                await self._random_delay()

            await browser.close()

        # 최종 저장
        self._save_final(store_prices, all_menus)

        print(f"\n{'='*60}")
        print(f"크롤링 완료!")
        print(f"{'='*60}")
        print(f"  성공: {success}개")
        print(f"  실패: {fail}개")
        print(f"  매칭률: {success/(success+fail)*100:.1f}%")

        return store_prices, all_menus

    # ──────────────────────────────────────────
    # 저장
    # ──────────────────────────────────────────
    def _save_intermediate(self, store_prices, all_menus, n):
        self._save_csv(store_prices, os.path.join(OUTPUT_DIR, "naver_prices_intermediate.csv"))
        self._save_csv(all_menus, os.path.join(OUTPUT_DIR, "naver_menus_intermediate.csv"))
        print(f"\n    [중간저장] {n}개 처리 완료")

    def _save_final(self, store_prices, all_menus):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        prices_file = os.path.join(OUTPUT_DIR, f"naver_prices_{ts}.csv")
        menus_file = os.path.join(OUTPUT_DIR, f"naver_menus_{ts}.csv")

        self._save_csv(store_prices, prices_file)
        self._save_csv(all_menus, menus_file)

        # JSON 백업
        json_file = os.path.join(OUTPUT_DIR, f"naver_data_{ts}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"store_prices": store_prices, "menus": all_menus}, f, ensure_ascii=False, indent=2)

        print(f"  가격 요약: {prices_file}")
        print(f"  메뉴 상세: {menus_file}")
        print(f"  JSON: {json_file}")

    @staticmethod
    def _save_csv(data, filepath):
        if not data:
            return
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)


# ──────────────────────────────────────────────
# 병합 스크립트
# ──────────────────────────────────────────────
def merge_google_naver(google_csv: str, naver_prices_csv: str, output_path: str = None):
    """
    Google Places 데이터에 네이버 메뉴 가격 병합

    결과:
      - price_median: 메뉴 가격 중앙값 (네이버)
      - price_mean: 메뉴 가격 평균 (네이버)
      - price_category: 가격대 범주 (중앙값 기반)
    """
    google_df = pd.read_csv(google_csv)
    naver_df = pd.read_csv(naver_prices_csv)

    # 병합 (google_place_id 기준)
    merged = google_df.merge(
        naver_df[["google_place_id", "naver_place_id", "menu_count",
                  "price_median", "price_mean", "price_min", "price_max", "status"]],
        left_on="place_id",
        right_on="google_place_id",
        how="left",
    )

    # 가격대 범주화 (중앙값 기준, 사분위수 또는 절대 기준)
    # 절대 기준: ~10,000원 저가 / ~25,000원 중가 / 25,000원+ 고가
    def categorize(price):
        if pd.isna(price):
            return None
        elif price <= 10000:
            return "저가"
        elif price <= 25000:
            return "중가"
        else:
            return "고가"

    merged["naver_price_category"] = merged["price_median"].apply(categorize)

    print(f"병합 결과: {len(merged)}행")
    print(f"  네이버 가격 있음: {merged['price_median'].notna().sum()}개")
    print(f"  네이버 가격 없음: {merged['price_median'].isna().sum()}개")
    print(f"\n가격대 분포:")
    print(merged["naver_price_category"].value_counts(dropna=False))

    if output_path:
        merged.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n저장: {output_path}")

    return merged


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("네이버 지도 메뉴 가격 크롤러")
    print("=" * 60)

    print("\n실행 모드:")
    print("  1) 테스트 (10개만)")
    print("  2) 소규모 (100개)")
    print("  3) 전체 크롤링")
    print("  4) 병합만 (이미 크롤링 완료된 경우)")

    mode = input("\n선택 (1/2/3/4): ").strip()

    if mode in ["1", "2", "3"]:
        csv_path = input("Google Places CSV 경로 (예: raw_crawled_data.csv): ").strip()
        if not csv_path:
            csv_path = "raw_crawled_data.csv"

        max_items = {
            "1": 10,
            "2": 100,
            "3": None,
        }[mode]

        crawler = NaverMenuCrawler(headless=False)
        await crawler.crawl_menu_prices(csv_path, max_items=max_items)

    elif mode == "4":
        google_csv = input("Google CSV 경로: ").strip()
        naver_csv = input("네이버 가격 CSV 경로: ").strip()
        output_path = input("출력 파일 경로 (예: merged_data.csv): ").strip()
        merge_google_naver(google_csv, naver_csv, output_path)

    else:
        print("잘못된 입력")


if __name__ == "__main__":
    asyncio.run(main())