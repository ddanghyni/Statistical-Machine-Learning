"""
크롤링 데이터 전처리 스크립트
==============================
- 메뉴 가격 중앙값 계산 → 가게 테이블에 병합
- 행정구 추출 및 정리
- 결측치 처리
- 가격대 범주화 (저가/중가/고가)
- 분석용 최종 데이터셋 생성
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Optional


OUTPUT_DIR = "output"
PROCESSED_DIR = "processed"


def load_latest_data(directory: str = OUTPUT_DIR):
    """가장 최근 크롤링 데이터 로드"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 가장 최근 파일 찾기
    rest_files = sorted(glob.glob(os.path.join(directory, "restaurants_2*.csv")))
    menu_files = sorted(glob.glob(os.path.join(directory, "menus_2*.csv")))

    if not rest_files:
        # intermediate 파일 확인
        rest_files = glob.glob(os.path.join(directory, "restaurants_intermediate.csv"))
        menu_files = glob.glob(os.path.join(directory, "menus_intermediate.csv"))

    if not rest_files:
        raise FileNotFoundError(f"{directory}에 크롤링 데이터가 없습니다.")

    rest_file = rest_files[-1]
    menu_file = menu_files[-1] if menu_files else None

    print(f"가게 데이터: {rest_file}")
    print(f"메뉴 데이터: {menu_file}")

    restaurants = pd.read_csv(rest_file)
    menus = pd.read_csv(menu_file) if menu_file else pd.DataFrame()

    print(f"\n원본 데이터:")
    print(f"  가게: {len(restaurants)}개")
    print(f"  메뉴: {len(menus)}개")

    return restaurants, menus


def compute_price_stats(menus: pd.DataFrame) -> pd.DataFrame:
    """
    가게별 메뉴 가격 통계 계산

    Returns:
        place_id별 가격 중앙값, 평균, 최소, 최대, 메뉴 수
    """
    if menus.empty or "price" not in menus.columns:
        return pd.DataFrame()

    # 가격이 있는 메뉴만
    menus_with_price = menus.dropna(subset=["price"])
    menus_with_price = menus_with_price[menus_with_price["price"] > 0]

    if menus_with_price.empty:
        return pd.DataFrame()

    price_stats = (
        menus_with_price.groupby("place_id")["price"]
        .agg(
            price_median="median",
            price_mean="mean",
            price_min="min",
            price_max="max",
            price_std="std",
            menu_with_price_count="count",
        )
        .reset_index()
    )

    # 반올림
    price_stats["price_median"] = price_stats["price_median"].round(0).astype(int)
    price_stats["price_mean"] = price_stats["price_mean"].round(0).astype(int)
    price_stats["price_std"] = price_stats["price_std"].round(0)

    return price_stats


def categorize_price(price_median: float) -> str:
    """
    가격대 범주화 (분석용)

    기준 (조정 가능):
        저가: ~10,000원
        중가: 10,000원 ~ 25,000원
        고가: 25,000원~
    """
    if pd.isna(price_median):
        return "정보없음"
    elif price_median <= 10000:
        return "저가"
    elif price_median <= 25000:
        return "중가"
    else:
        return "고가"


def categorize_business_type(category: str) -> str:
    """업종 대분류"""
    if pd.isna(category):
        return "기타"

    category = str(category)

    cafe_keywords = ["카페", "커피", "디저트", "베이커리", "빵", "케이크", "차"]
    for kw in cafe_keywords:
        if kw in category:
            return "카페"

    return "음식점"


def preprocess(restaurants: pd.DataFrame, menus: pd.DataFrame) -> pd.DataFrame:
    """
    전처리 메인 함수

    1. 폐업 가게 제거
    2. 메뉴 가격 통계 병합
    3. 가격대 범주화
    4. 업종 대분류
    5. 리뷰 수 합산
    6. 결측치 정리
    """
    df = restaurants.copy()
    print(f"\n{'='*50}")
    print("전처리 시작")
    print(f"{'='*50}")

    # 1) 폐업 가게 제거
    before = len(df)
    if "business_status" in df.columns:
        df = df[df["business_status"] == "영업중"]
    print(f"\n[1] 폐업/휴업 제거: {before} → {len(df)}개")

    # 2) 메뉴 가격 통계 병합
    price_stats = compute_price_stats(menus)
    if not price_stats.empty:
        df = df.merge(price_stats, on="place_id", how="left")
        print(f"[2] 가격 정보 병합 완료 (가격 있는 가게: {price_stats.shape[0]}개)")
    else:
        df["price_median"] = np.nan
        df["price_mean"] = np.nan
        print(f"[2] 메뉴 가격 데이터 없음")

    # 3) 가격대 범주화
    df["price_category"] = df["price_median"].apply(categorize_price)
    print(f"[3] 가격대 범주화 완료")
    print(f"    {df['price_category'].value_counts().to_dict()}")

    # 4) 업종 대분류
    df["business_type"] = df["category"].apply(categorize_business_type)
    print(f"[4] 업종 분류 완료")
    print(f"    {df['business_type'].value_counts().to_dict()}")

    # 5) 총 리뷰 수
    df["total_review_count"] = (
        df["visitor_review_count"].fillna(0) + df["blog_review_count"].fillna(0)
    )
    print(f"[5] 총 리뷰 수 계산 완료")

    # 6) 평점 없는 가게 처리
    before = len(df)
    df_with_rating = df.dropna(subset=["rating"])
    print(f"[6] 평점 없는 가게: {before - len(df_with_rating)}개")

    # 7) 기본 통계
    print(f"\n{'='*50}")
    print("전처리 완료 요약")
    print(f"{'='*50}")
    print(f"  전체 가게 수: {len(df)}개")
    print(f"  평점 있는 가게: {len(df_with_rating)}개")
    print(f"  가격 있는 가게: {df['price_median'].notna().sum()}개")
    print(f"  평점 범위: {df['rating'].min()} ~ {df['rating'].max()}")
    print(f"  평점 평균: {df['rating'].mean():.2f}")
    if df["price_median"].notna().any():
        print(f"  가격 중앙값 범위: {df['price_median'].min():.0f} ~ {df['price_median'].max():.0f}원")
    print(f"  행정구: {df['district'].nunique()}개")
    print(f"  업종: {df['category'].nunique()}개")

    return df


def save_processed(df: pd.DataFrame, menus: pd.DataFrame):
    """전처리된 데이터 저장"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 전체 데이터 (평점 없는 것 포함)
    all_file = os.path.join(PROCESSED_DIR, "all_stores.csv")
    df.to_csv(all_file, index=False, encoding="utf-8-sig")

    # 분석용 데이터 (평점 있는 것만)
    analysis_df = df.dropna(subset=["rating"])
    analysis_file = os.path.join(PROCESSED_DIR, "analysis_data.csv")
    analysis_df.to_csv(analysis_file, index=False, encoding="utf-8-sig")

    # 메뉴 데이터
    menu_file = os.path.join(PROCESSED_DIR, "menus_clean.csv")
    menus.to_csv(menu_file, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료:")
    print(f"  전체 데이터:  {all_file} ({len(df)}개)")
    print(f"  분석용 데이터: {analysis_file} ({len(analysis_df)}개)")
    print(f"  메뉴 데이터:  {menu_file} ({len(menus)}개)")


def main():
    # 1) 데이터 로드
    restaurants, menus = load_latest_data()

    # 2) 전처리
    df = preprocess(restaurants, menus)

    # 3) 저장
    save_processed(df, menus)

    print("\n전처리 완료! 'processed/' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
