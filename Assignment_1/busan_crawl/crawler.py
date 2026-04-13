import requests
import csv
import json
import os
import time
import re
from datetime import datetime
from typing import List, Dict, Optional

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
OUTPUT_DIR = "output"

# 부산 행정구별 중심 좌표 (위도, 경도)
BUSAN_DISTRICT_COORDS = {
    "해운대구": (35.1631, 129.1635),
    "부산진구": (35.1631, 129.0532),
    "금정구":   (35.2437, 129.0922),
    "남구":     (35.1368, 129.0843),
    "수영구":   (35.1457, 129.1133),
    "중구":     (35.1064, 129.0324),
    "동래구":   (35.2050, 129.0858),
    "사하구":   (35.1047, 128.9748),
    "북구":     (35.1972, 129.0312),
    "사상구":   (35.1526, 128.9910),
    "연제구":   (35.1762, 129.0801),
    "영도구":   (35.0911, 129.0678),
    "강서구":   (35.1122, 128.8723),
    "동구":     (35.1295, 129.0454),
    "서구":     (35.0977, 129.0243),
    "기장군":   (35.2446, 129.2222),
}

# 검색할 장소 유형
PLACE_TYPES = {
    "음식점": ["restaurant"],
    "카페": ["cafe"],
}

# API 요청 간 딜레이 (초)
REQUEST_DELAY = 0.5


class GooglePlacesCollector:
    """Google Places API (New) 데이터 수집기"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.crawl_date = datetime.now().strftime("%Y-%m-%d")
        self.collected_place_ids = set()
        self.all_restaurants = []
        self.request_count = 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────
    # Nearby Search (New)
    # ──────────────────────────────────────────
    def nearby_search(
        self,
        lat: float,
        lng: float,
        radius: float = 2000,
        place_types: List[str] = None,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        좌표 기반 주변 장소 검색

        Args:
            lat, lng: 중심 좌표
            radius: 검색 반경 (미터)
            place_types: 장소 유형 (예: ["restaurant", "cafe"])
            max_results: 최대 결과 수 (1~20)
        """
        url = "https://places.googleapis.com/v1/places:searchNearby"

        # 요청할 필드 (비용 최적화)
        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.primaryType",
            "places.primaryTypeDisplayName",
            "places.types",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.formattedAddress",
            "places.location",
            "places.businessStatus",
        ])

        body = {
            "maxResultCount": min(max_results, 20),
            "languageCode": "ko",
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": radius,
                }
            },
        }

        if place_types:
            body["includedTypes"] = place_types

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        try:
            resp = requests.post(url, json=body, headers=headers)
            self.request_count += 1

            if resp.status_code != 200:
                print(f"    [!] API 오류 ({resp.status_code}): {resp.text[:200]}")
                return []

            data = resp.json()
            return data.get("places", [])

        except Exception as e:
            print(f"    [!] 요청 실패: {e}")
            return []

    # ──────────────────────────────────────────
    # Text Search (New) - 키워드 기반 검색
    # ──────────────────────────────────────────
    def text_search(
        self,
        query: str,
        lat: float = None,
        lng: float = None,
        radius: float = 5000,
        max_results: int = 20,
    ) -> List[Dict]:
        """
        키워드 기반 장소 검색

        Args:
            query: 검색어 (예: "부산 해운대 맛집")
            lat, lng: 검색 중심 좌표 (선택)
            radius: 검색 반경
        """
        url = "https://places.googleapis.com/v1/places:searchText"

        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.primaryType",
            "places.primaryTypeDisplayName",
            "places.types",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.formattedAddress",
            "places.location",
            "places.businessStatus",
        ])

        body = {
            "textQuery": query,
            "maxResultCount": min(max_results, 20),
            "languageCode": "ko",
        }

        if lat and lng:
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": radius,
                }
            }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        try:
            resp = requests.post(url, json=body, headers=headers)
            self.request_count += 1

            if resp.status_code != 200:
                print(f"    [!] API 오류 ({resp.status_code}): {resp.text[:200]}")
                return []

            data = resp.json()
            return data.get("places", [])

        except Exception as e:
            print(f"    [!] 요청 실패: {e}")
            return []

    # ──────────────────────────────────────────
    # 결과 파싱
    # ──────────────────────────────────────────
    def _parse_place(self, place: Dict, search_query: str = "") -> Optional[Dict]:
        """API 응답을 우리 데이터 형식으로 변환"""
        place_id = place.get("id", "")

        # 중복 체크
        if place_id in self.collected_place_ids:
            return None
        self.collected_place_ids.add(place_id)

        # 이름
        display_name = place.get("displayName", {})
        name = display_name.get("text", "") if isinstance(display_name, dict) else str(display_name)

        # 업종
        primary_type = place.get("primaryType", "")
        primary_type_display = place.get("primaryTypeDisplayName", {})
        category = primary_type_display.get("text", primary_type) if isinstance(primary_type_display, dict) else str(primary_type_display)

        # 평점 (1.0 ~ 5.0)
        rating = place.get("rating")

        # 리뷰 수
        review_count = place.get("userRatingCount", 0)

        # 가격대 (PRICE_LEVEL_FREE=0 ~ PRICE_LEVEL_VERY_EXPENSIVE=4)
        price_level_raw = place.get("priceLevel", "")
        price_level = self._parse_price_level(price_level_raw)

        # 주소
        address = place.get("formattedAddress", "")

        # 좌표
        location = place.get("location", {})
        lat = location.get("latitude")
        lng = location.get("longitude")

        # 영업 상태
        business_status = place.get("businessStatus", "OPERATIONAL")

        # 행정구 추출
        district = self._extract_district(address)

        # types 리스트
        types = place.get("types", [])

        # 업종 대분류
        business_type = "카페" if any(t in types for t in ["cafe", "coffee_shop"]) else "음식점"

        return {
            "place_id": place_id,
            "name": name,
            "category": category,
            "business_type": business_type,
            "rating": rating,
            "review_count": review_count,
            "price_level": price_level,
            "price_level_raw": price_level_raw,
            "address": address,
            "district": district,
            "latitude": lat,
            "longitude": lng,
            "business_status": business_status,
            "types": ",".join(types),
            "search_query": search_query,
            "crawl_date": self.crawl_date,
        }

    @staticmethod
    def _parse_price_level(raw) -> Optional[int]:
        """가격대 문자열을 숫자로 변환"""
        mapping = {
            "PRICE_LEVEL_FREE": 0,
            "PRICE_LEVEL_INEXPENSIVE": 1,
            "PRICE_LEVEL_MODERATE": 2,
            "PRICE_LEVEL_EXPENSIVE": 3,
            "PRICE_LEVEL_VERY_EXPENSIVE": 4,
        }
        if isinstance(raw, str):
            return mapping.get(raw)
        if isinstance(raw, (int, float)):
            return int(raw)
        return None

    @staticmethod
    def _extract_district(address: str) -> str:
        if not address:
            return ""
        m = re.search(r"부산\s*(광역시)?\s*(\S+[구군])", address)
        return m.group(2) if m else ""

    # ──────────────────────────────────────────
    # 수집 전략: 행정구별 격자 + Nearby Search
    # ──────────────────────────────────────────
    def collect_by_grid(
        self,
        district: str,
        center_lat: float,
        center_lng: float,
        place_types: List[str],
        type_label: str,
        radius: float = 1500,
        grid_offsets: List[tuple] = None,
    ):
        """
        한 행정구를 격자로 나눠서 Nearby Search 여러 번 호출
        → 한 번에 최대 20개만 반환하므로, 좌표를 조금씩 이동시켜 더 많이 수집
        5×5 격자 (25개 포인트) + 반경 1.5km → 행정구 대부분 커버
        """
        if grid_offsets is None:
            # 5×5 격자 (25개 포인트)
            delta = 0.012  # 약 1.2km
            offsets = [-2*delta, -delta, 0, delta, 2*delta]
            grid_offsets = [(dlat, dlng) for dlat in offsets for dlng in offsets]

        total_new = 0

        for i, (dlat, dlng) in enumerate(grid_offsets):
            search_lat = center_lat + dlat
            search_lng = center_lng + dlng

            places = self.nearby_search(
                lat=search_lat,
                lng=search_lng,
                radius=radius,
                place_types=place_types,
                max_results=20,
            )

            new_count = 0
            for place in places:
                parsed = self._parse_place(
                    place,
                    search_query=f"{district} {type_label}",
                )
                if parsed:
                    # 영업중인 곳만
                    if parsed["business_status"] == "OPERATIONAL":
                        self.all_restaurants.append(parsed)
                        new_count += 1

            total_new += new_count
            time.sleep(REQUEST_DELAY)

        return total_new

    def collect_by_text_search(
        self,
        district: str,
        center_lat: float,
        center_lng: float,
        keyword: str,
    ):
        """Text Search로 수집 (Nearby Search 보완)"""
        query = f"부산 {district} {keyword}"

        places = self.text_search(
            query=query,
            lat=center_lat,
            lng=center_lng,
            radius=5000,
            max_results=20,
        )

        new_count = 0
        for place in places:
            parsed = self._parse_place(place, search_query=query)
            if parsed and parsed["business_status"] == "OPERATIONAL":
                self.all_restaurants.append(parsed)
                new_count += 1

        time.sleep(REQUEST_DELAY)
        return new_count

    # ──────────────────────────────────────────
    # 전체 수집
    # ──────────────────────────────────────────
    def collect_all(self, districts: Dict[str, tuple] = None):
        """부산 전체 데이터 수집"""
        if districts is None:
            districts = BUSAN_DISTRICT_COORDS

        print("=" * 60)
        print("부산 카페/음식점 Google Places API 데이터 수집")
        print("=" * 60)
        print(f"대상 행정구: {len(districts)}개")
        print(f"수집 시작: {self.crawl_date}")

        total_districts = len(districts)

        for idx, (district, (lat, lng)) in enumerate(districts.items(), 1):
            print(f"\n[{idx}/{total_districts}] {district} (중심: {lat}, {lng})")

            # 1) Nearby Search: 음식점
            n1 = self.collect_by_grid(
                district, lat, lng,
                place_types=["restaurant"],
                type_label="음식점",
            )
            print(f"  음식점 (Nearby): +{n1}개")

            # 2) Nearby Search: 카페
            n2 = self.collect_by_grid(
                district, lat, lng,
                place_types=["cafe"],
                type_label="카페",
            )
            print(f"  카페 (Nearby): +{n2}개")

            # 3) Text Search 보완 (더 다양한 키워드로)
            text_new = 0
            for keyword in ["맛집", "카페", "음식점", "식당", "커피"]:
                n = self.collect_by_text_search(district, lat, lng, keyword)
                text_new += n
            print(f"  Text Search 보완: +{text_new}개")

            print(f"  → 누적: {len(self.all_restaurants)}개 (API 요청: {self.request_count}회)")

        # 저장
        self._save_results()

    # ──────────────────────────────────────────
    # 저장
    # ──────────────────────────────────────────
    def _save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_file = os.path.join(OUTPUT_DIR, f"google_places_{ts}.csv")
        self._save_csv(self.all_restaurants, csv_file)

        # JSON
        json_file = os.path.join(OUTPUT_DIR, f"google_places_{ts}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.all_restaurants, f, ensure_ascii=False, indent=2)

        # 통계
        print(f"\n{'='*60}")
        print(f"수집 완료!")
        print(f"{'='*60}")
        print(f"  총 가게 수: {len(self.all_restaurants)}개")
        print(f"  고유 place_id: {len(self.collected_place_ids)}개")
        print(f"  API 요청 횟수: {self.request_count}회")

        # 기본 통계
        ratings = [r["rating"] for r in self.all_restaurants if r["rating"]]
        prices = [r["price_level"] for r in self.all_restaurants if r["price_level"] is not None]
        reviews = [r["review_count"] for r in self.all_restaurants if r["review_count"]]

        if ratings:
            print(f"  평점 있는 가게: {len(ratings)}개 (평균: {sum(ratings)/len(ratings):.2f})")
        if prices:
            print(f"  가격대 있는 가게: {len(prices)}개")
        if reviews:
            print(f"  리뷰 있는 가게: {len(reviews)}개 (평균: {sum(reviews)/len(reviews):.0f}건)")

        districts = set(r["district"] for r in self.all_restaurants if r["district"])
        print(f"  행정구: {len(districts)}개 → {', '.join(sorted(districts))}")

        print(f"\n  CSV: {csv_file}")
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
# 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("부산 카페/음식점 Google Places API 수집기")
    print("=" * 60)

    # API 키 입력
    api_key = input("\nGoogle Places API 키를 입력하세요: ").strip()
    if not api_key:
        print("API 키가 필요합니다.")
        return

    collector = GooglePlacesCollector(api_key)

    print("\n실행 모드:")
    print("  1) 전체 수집 (16개 구·군)")
    print("  2) 특정 지역만")
    print("  3) API 테스트 (해운대구 1개만)")

    mode = input("\n선택 (1/2/3): ").strip()

    if mode == "1":
        collector.collect_all()

    elif mode == "2":
        print("\n부산 행정구:")
        districts_list = list(BUSAN_DISTRICT_COORDS.keys())
        for i, d in enumerate(districts_list, 1):
            print(f"  {i:2d}) {d}")
        sel = input("\n번호 (쉼표 구분): ").strip()
        indices = [int(x.strip()) - 1 for x in sel.split(",")]
        selected = {
            districts_list[i]: BUSAN_DISTRICT_COORDS[districts_list[i]]
            for i in indices
            if 0 <= i < len(districts_list)
        }
        print(f"\n선택: {', '.join(selected.keys())}")
        collector.collect_all(districts=selected)

    elif mode == "3":
        print("\n[API 테스트] 해운대구 음식점 20개만 검색...")
        places = collector.nearby_search(
            lat=35.1631, lng=129.1635,
            radius=2000,
            place_types=["restaurant"],
            max_results=20,
        )
        print(f"\n검색 결과: {len(places)}개")
        for i, p in enumerate(places, 1):
            name = p.get("displayName", {}).get("text", "?")
            rating = p.get("rating", "N/A")
            count = p.get("userRatingCount", 0)
            price = p.get("priceLevel", "N/A")
            addr = p.get("formattedAddress", "")[:40]
            print(f"  {i:2d}. {name} ★{rating} ({count}건) 가격:{price} | {addr}")

        print(f"\nAPI 요청 횟수: {collector.request_count}회")
        print("테스트 성공! 전체 수집은 모드 1로 실행하세요.")

    else:
        print("잘못된 입력")


if __name__ == "__main__":
    main()