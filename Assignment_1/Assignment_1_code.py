# %% 0. 환경 설정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 (환경에 맞게 수정) ──
# Windows: "Malgun Gothic"
# Mac: "AppleGothic"  
# Linux: "Noto Sans CJK KR"
#plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["font.family"] = "AppleGothic" 
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 시각화 스타일
sns.set_style("whitegrid")
sns.set_palette("Set2")

print("환경 설정 완료")
# %% 1. 데이터 로드
# 원본 데이터 로드
df_raw = pd.read_csv("/Users/sanghyun/Desktop/code_attic/output/google_places_20260317_090717.csv") 

print(f"원본 데이터: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")
print(f"\n컬럼 목록:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col:<20s} (dtype: {df_raw[col].dtype})")
# %%
# ── 2.1 분석 대상 필터링 ──

# 부산 16개 행정구·군
valid_districts = [
    "해운대구", "부산진구", "금정구", "남구", "수영구",
    "중구", "동래구", "사하구", "북구", "사상구",
    "연제구", "영도구", "강서구", "동구", "서구", "기장군"
]

# 조건: (1) price_level 존재 (2) rating 존재 (3) 유효 행정구
df = df_raw[
    df_raw["price_level"].notna() &
    df_raw["rating"].notna() &
    df_raw["district"].isin(valid_districts)
].copy()

print(f"필터링 결과:")
print(f"  원본: {len(df_raw)}개")
print(f"  → price_level 존재 + rating 존재 + 유효 행정구: {len(df)}개")
print(f"  제거된 행: {len(df_raw) - len(df)}개")
# %%
# ── 2.2 결측치 확인 ──
print("필터링 후 결측치 현황:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  → 주요 분석 변수에 결측치 없음")

print(f"\n각 변수별 유효 데이터 수:")
print(f"  rating:       {df['rating'].notna().sum()}")
print(f"  review_count: {df['review_count'].notna().sum()}")
print(f"  price_level:  {df['price_level'].notna().sum()}")
print(f"  district:     {df['district'].notna().sum()}")
print(f"  business_type:{df['business_type'].notna().sum()}")
# %%
# ── 2.3-1 price_level 원본 상태 확인 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) price_level 원본 분포 
price_raw_counts = df["price_level"].value_counts().sort_index()
labels_raw = {1.0: "1\n(INEXPENSIVE)", 2.0: "2\n(MODERATE)", 3.0: "3\n(EXPENSIVE)", 4.0: "4\n(VERY_EXPENSIVE)"}
colors_raw = ["#70AD47", "#5B9BD5", "#ED7D31", "#C00000"]
bars = axes[0].bar([labels_raw[k] for k in price_raw_counts.index], price_raw_counts.values, 
                    color=colors_raw, edgecolor="white")
for bar, val in zip(bars, price_raw_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}개\n({val/len(df)*100:.1f}%)", 
                 ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylabel("가게 수", fontsize=11)
axes[0].set_title("(a) price_level 원본 분포\n→ 4(VERY_EXPENSIVE)가 5개뿐", fontsize=12)

# (우) 범주화 후 — 고가(3) + 최고가(4)를 합침
# 먼저 범주화 수행
def categorize_price(level):
    if level == 1:
        return "저가"
    elif level == 2:
        return "중가"
    else:  # 3, 4
        return "고가"

df["price_category"] = df["price_level"].apply(categorize_price)
price_cat_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
colors_cat = ["#70AD47", "#5B9BD5", "#ED7D31"]
bars = axes[1].bar(price_cat_counts.index, price_cat_counts.values, color=colors_cat, edgecolor="white")
for bar, val in zip(bars, price_cat_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}개\n({val/len(df)*100:.1f}%)", 
                 ha="center", fontsize=10, fontweight="bold")
axes[1].set_ylabel("가게 수", fontsize=11)
axes[1].set_title("(b) 범주화 후: 고가 + 최고가 합침\n→ 3그룹으로 비교 가능", fontsize=12)

plt.suptitle("price_level 변환: 원본 → 범주화", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# ── 2.3-1b Class Imbalance 수치 비교 ──
# 합치기 전후로 클래스 불균형이 얼마나 개선되었는지 정량적으로 확인

n = len(df)

# 합치기 전 (4 클래스)
before_counts = df["price_level"].value_counts().sort_index()
props_before = before_counts / n
ir_before = before_counts.max() / before_counts.min()  # Imbalance Ratio
entropy_before = -np.sum(props_before * np.log(props_before))
balance_before = entropy_before / np.log(len(before_counts))  # 1이면 완전 균형

# 합친 후 (3 클래스)
after_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
props_after = after_counts / n
ir_after = after_counts.max() / after_counts.min()
entropy_after = -np.sum(props_after * np.log(props_after))
balance_after = entropy_after / np.log(len(after_counts))

# 비교 테이블
print("Class Imbalance 비교 (합치기 전 vs 후)")
print(f"{'─'*65}")
print(f"{'지표':<30} {'합치기 전(4cls)':<18} {'합친 후(3cls)':<15}")
print(f"{'─'*65}")
print(f"{'클래스 수':<30} {len(before_counts):<18} {len(after_counts):<15}")
print(f"{'최소 클래스 크기':<30} {before_counts.min():<18} {after_counts.min():<15}")
print(f"{'최대/최소 비율 (IR)':<30} {ir_before:<18.1f} {ir_after:<15.1f}")
print(f"{'Balance Ratio (0~1)':<30} {balance_before:<18.4f} {balance_after:<15.4f}")
print(f"{'─'*65}")
print(f"  IR: {ir_before:.0f}:1 → {ir_after:.0f}:1 (↓ 개선)")
print(f"  Balance Ratio: {balance_before:.3f} → {balance_after:.3f} (↑ 개선)")

# 시각화: 비교 막대그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

metrics = ["Imbalance Ratio\n(최대/최소, 낮을수록 균형)", "Balance Ratio\n(0~1, 높을수록 균형)"]
before_vals = [ir_before, balance_before]
after_vals = [ir_after, balance_after]

x = np.arange(len(metrics))
w = 0.3
bars1 = axes[0].bar(x - w/2, before_vals, w, label="합치기 전 (4cls)", color="#C44E52", alpha=0.8)
bars2 = axes[0].bar(x + w/2, after_vals, w, label="합친 후 (3cls)", color="#5B9BD5", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=10)
axes[0].set_ylabel("값", fontsize=11)
axes[0].set_title("(a) 불균형 지표 비교", fontsize=12)
axes[0].legend()
# 값 표시
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{bar.get_height():.1f}", ha="center", fontsize=9)

# 비율 파이차트 (합친 후)
axes[1].pie(after_counts.values, labels=[f"{k}\n({v}개, {v/n*100:.1f}%)" for k, v in after_counts.items()],
            colors=["#70AD47", "#5B9BD5", "#ED7D31"], autopct="", startangle=90, 
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("(b) 합친 후 비율 분포\n→ 중가 74.8% 여전히 지배적", fontsize=12)

plt.suptitle("Class Imbalance 분석", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\n※ 합친 후에도 중가(74.8%)가 지배적인 class imbalance가 존재함")
print(f"  → 소주제 1 분석 시 비모수 검정(Kruskal-Wallis) 사용 및 해석에 주의 필요")

print(f"\n결정: VERY_EXPENSIVE(4)가 5개뿐이므로 EXPENSIVE(3)와 합쳐 '고가'로 범주화")
print(f"  저가 (1): {(df['price_category']=='저가').sum()}개")
print(f"  중가 (2): {(df['price_category']=='중가').sum()}개")
print(f"  고가 (3+4): {(df['price_category']=='고가').sum()}개")

# %%
# ── 2.3-2 review_count 원본 상태 확인 + 로그 변환 ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# (좌) 원본 분포 — 극단적 right-skewed
axes[0].hist(df["review_count"], bins=50, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0].axvline(df["review_count"].mean(), color="red", linestyle="--", 
                label=f'평균: {df["review_count"].mean():.0f}')
axes[0].axvline(df["review_count"].median(), color="orange", linestyle="--", 
                label=f'중앙값: {df["review_count"].median():.0f}')
axes[0].set_xlabel("리뷰 수", fontsize=11)
axes[0].set_ylabel("빈도", fontsize=11)
axes[0].set_title(f"(a) 원본 분포\n왜도(skewness) = {df['review_count'].skew():.2f}", fontsize=12)
axes[0].legend(fontsize=9)

# (중) 로그 변환 후 — 훨씬 대칭적
df["log_review_count"] = np.log1p(df["review_count"])
axes[1].hist(df["log_review_count"], bins=30, color="#70AD47", edgecolor="white", alpha=0.85)
axes[1].axvline(df["log_review_count"].mean(), color="red", linestyle="--", 
                label=f'평균: {df["log_review_count"].mean():.2f}')
axes[1].axvline(df["log_review_count"].median(), color="orange", linestyle="--", 
                label=f'중앙값: {df["log_review_count"].median():.2f}')
axes[1].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[1].set_ylabel("빈도", fontsize=11)
axes[1].set_title(f"(b) 로그 변환 후\n왜도(skewness) = {df['log_review_count'].skew():.2f}", fontsize=12)
axes[1].legend(fontsize=9)

# (우) Q-Q plot으로 정규성 비교
from scipy.stats import probplot
probplot(df["log_review_count"], dist="norm", plot=axes[2])
axes[2].set_title("(c) 로그 변환 후 Q-Q Plot\n→ 정규분포에 근접", fontsize=12)

plt.suptitle("review_count 변환: 원본 → log(1+x)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"변환 전: 왜도 = {df['review_count'].skew():.2f} (심한 right-skew)")
print(f"변환 후: 왜도 = {df['log_review_count'].skew():.2f} (대칭에 가까움)")
print(f"→ 회귀분석 등에서 log_review_count를 사용하면 모델 가정에 더 적합")

# %%
# ── 2.3-3 리뷰 수 구간 변수 생성 ──
df["review_group"] = pd.cut(
    df["review_count"],
    bins=[0, 50, 100, 300, 1000, np.inf],
    labels=["~50", "51~100", "101~300", "301~1000", "1000+"],
    include_lowest=True,
)

fig, ax = plt.subplots(figsize=(8, 4))
rg_counts = df["review_group"].value_counts().sort_index()
bars = ax.bar(rg_counts.index, rg_counts.values, color="#5B9BD5", edgecolor="white")
for bar, val in zip(bars, rg_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 5, str(val), ha="center", fontweight="bold")
ax.set_xlabel("리뷰 수 구간", fontsize=11)
ax.set_ylabel("가게 수", fontsize=11)
ax.set_title("리뷰 수 구간별 가게 수", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("리뷰 수 구간별 분포:")
print(rg_counts.to_string())
# %%
# 수치형 독립변수 간 상관계수
indep_vars = ["price_level", "review_count", "log_review_count"]
indep_corr = df[indep_vars].corr()

print("독립변수 간 Pearson 상관계수:")
print(indep_corr.round(3).to_string())

print(f"\n주요 쌍별 상관:")
print(f"  price_level ↔ review_count:     r = {df['price_level'].corr(df['review_count']):.3f}")
print(f"  price_level ↔ log_review_count: r = {df['price_level'].corr(df['log_review_count']):.3f}")

# review_count와 log_review_count는 변환 관계이므로 당연히 높음 → 둘 중 하나만 모델에 사용
print(f"  review_count ↔ log_review_count: r = {df['review_count'].corr(df['log_review_count']):.3f} (변환 관계, 둘 중 하나만 사용)")

print(f"\n판단:")
print(f"  price_level ↔ log_review_count: |r| = {abs(df['price_level'].corr(df['log_review_count'])):.3f} < 0.3")
print(f"  → 독립변수 간 심각한 다중공선성 없음")
print(f"  → 소주제 5에서 VIF로 정식 진단 예정")
# %%
# ── 최종 데이터 요약 ──
print("=" * 55)
print("전처리 완료 데이터 요약")
print("=" * 55)
print(f"  총 관측치:     {len(df)}개")
print(f"  행정구:        {df['district'].nunique()}개")
print(f"  업종:          음식점 {(df['business_type']=='음식점').sum()}개 / 카페 {(df['business_type']=='카페').sum()}개")
print(f"")
print(f"  [종속변수]")
print(f"  rating:        {df['rating'].min()} ~ {df['rating'].max()} (평균: {df['rating'].mean():.3f}, 중앙값: {df['rating'].median()})")
print(f"")
print(f"  [독립변수]")
print(f"  price_category: 저가 {(df['price_category']=='저가').sum()} / 중가 {(df['price_category']=='중가').sum()} / 고가 {(df['price_category']=='고가').sum()}")
print(f"  review_count:   {df['review_count'].min()} ~ {df['review_count'].max()} (중앙값: {df['review_count'].median():.0f})")
print(f"  district:       {df['district'].nunique()}개 행정구 (최소 {df['district'].value_counts().min()}개 ~ 최대 {df['district'].value_counts().max()}개)")
print(f"  business_type:  음식점 / 카페")

# ── 분석에 사용할 변수 정리 ──
print(f"\n분석에 사용할 변수:")
print(f"  종속변수: rating (평점, 1.0~5.0)")
print(f"  독립변수: price_category, log_review_count, district, business_type")
print(f"  파생변수: price_category, log_review_count, review_group")
# %% 3.1 종속변수: 평점(rating) 분포
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (좌) 히스토그램
axes[0].hist(df["rating"], bins=25, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0].axvline(df["rating"].mean(), color="red", linestyle="--", linewidth=1.5, 
                label=f'평균: {df["rating"].mean():.2f}')
axes[0].axvline(df["rating"].median(), color="orange", linestyle="--", linewidth=1.5, 
                label=f'중앙값: {df["rating"].median():.1f}')
axes[0].set_xlabel("평점", fontsize=12)
axes[0].set_ylabel("빈도", fontsize=12)
axes[0].set_title("(a) 평점 분포", fontsize=13)
axes[0].legend(fontsize=10)

# (우) 박스플롯
bp = axes[1].boxplot(df["rating"], vert=True, patch_artist=True, 
                      boxprops=dict(facecolor="#5B9BD5", alpha=0.7))
axes[1].set_ylabel("평점", fontsize=12)
axes[1].set_title("(b) 평점 박스플롯", fontsize=13)
axes[1].set_xticklabels(["전체"])

# 통계량 텍스트
stats_text = (f'n = {len(df)}\n'
              f'평균 = {df["rating"].mean():.2f}\n'
              f'표준편차 = {df["rating"].std():.2f}\n'
              f'왜도 = {df["rating"].skew():.2f}')
axes[1].text(1.3, df["rating"].min() + 0.1, stats_text, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Figure 1. 종속변수(평점) 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %% 3.2 독립변수 분포
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) 가격대 분포
price_counts = df["price_category"].value_counts().reindex(["저가", "중가", "고가"])
colors_price = ["#70AD47", "#5B9BD5", "#ED7D31"]
bars = axes[0, 0].bar(price_counts.index, price_counts.values, color=colors_price, edgecolor="white")
for bar, val in zip(bars, price_counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha="center", fontweight="bold")
axes[0, 0].set_ylabel("가게 수", fontsize=11)
axes[0, 0].set_title("(a) 가격대별 가게 수", fontsize=12)

# (b) 리뷰 수 분포 (로그 스케일)
axes[0, 1].hist(df["log_review_count"], bins=30, color="#5B9BD5", edgecolor="white", alpha=0.85)
axes[0, 1].axvline(df["log_review_count"].median(), color="orange", linestyle="--", 
                    label=f'중앙값: {df["review_count"].median():.0f}건')
axes[0, 1].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[0, 1].set_ylabel("빈도", fontsize=11)
axes[0, 1].set_title("(b) 리뷰 수 분포 (로그 변환 후)", fontsize=12)
axes[0, 1].legend(fontsize=10)

# (c) 업종 분포
bt_counts = df["business_type"].value_counts()
colors_bt = ["#5B9BD5", "#ED7D31"]
bars = axes[1, 0].bar(bt_counts.index, bt_counts.values, color=colors_bt, edgecolor="white")
for bar, val in zip(bars, bt_counts.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 10, f"{val}\n({val/len(df)*100:.1f}%)", 
                     ha="center", fontsize=10)
axes[1, 0].set_ylabel("가게 수", fontsize=11)
axes[1, 0].set_title("(c) 업종별 가게 수", fontsize=12)

# (d) 행정구별 가게 수
dist_counts = df["district"].value_counts().sort_values()
axes[1, 1].barh(dist_counts.index, dist_counts.values, color="#5B9BD5", edgecolor="white")
for i, val in enumerate(dist_counts.values):
    axes[1, 1].text(val + 2, i, str(val), va="center", fontsize=9)
axes[1, 1].set_xlabel("가게 수", fontsize=11)
axes[1, 1].set_title("(d) 행정구별 가게 수", fontsize=12)

plt.suptitle("Figure 2. 독립변수 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %% 3.3 변수 간 상관관계
# 수치형 변수 간 상관행렬
corr_vars = ["rating", "review_count", "log_review_count", "price_level"]
corr_labels = ["평점", "리뷰 수", "log 리뷰 수", "가격대"]
corr_matrix = df[corr_vars].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, vmin=-1, vmax=1, ax=ax,
            xticklabels=corr_labels, yticklabels=corr_labels,
            linewidths=0.5)
ax.set_title("Figure 3. 수치형 변수 간 상관행렬 (Pearson)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 주요 상관계수 출력
print("주요 상관관계:")
print(f"  평점 ↔ 리뷰 수:     r = {corr_matrix.loc['rating','review_count']:.3f}")
print(f"  평점 ↔ log 리뷰 수: r = {corr_matrix.loc['rating','log_review_count']:.3f}")
print(f"  평점 ↔ 가격대:      r = {corr_matrix.loc['rating','price_level']:.3f}")
print(f"  리뷰 수 ↔ 가격대:   r = {corr_matrix.loc['review_count','price_level']:.3f}")
# %% 3.4 주요 이변량 관계 탐색
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) 가격대별 평점
order_price = ["저가", "중가", "고가"]
sns.boxplot(data=df, x="price_category", y="rating", order=order_price,
            palette=["#70AD47", "#5B9BD5", "#ED7D31"], ax=axes[0], width=0.5)
# 평균 표시
for i, cat in enumerate(order_price):
    mean_val = df[df["price_category"] == cat]["rating"].mean()
    axes[0].scatter(i, mean_val, color="red", s=60, zorder=5, marker="D")
    axes[0].text(i + 0.15, mean_val, f"{mean_val:.2f}", color="red", fontsize=10, fontweight="bold")
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 가격대별 평점", fontsize=12)

# (b) 업종별 평점
sns.boxplot(data=df, x="business_type", y="rating",
            palette=["#5B9BD5", "#ED7D31"], ax=axes[1], width=0.4)
for i, bt in enumerate(df["business_type"].unique()):
    mean_val = df[df["business_type"] == bt]["rating"].mean()
    axes[1].scatter(i, mean_val, color="red", s=60, zorder=5, marker="D")
    axes[1].text(i + 0.12, mean_val, f"{mean_val:.2f}", color="red", fontsize=10, fontweight="bold")
axes[1].set_xlabel("업종", fontsize=11)
axes[1].set_ylabel("평점", fontsize=11)
axes[1].set_title("(b) 업종별 평점", fontsize=12)

# (c) 리뷰 수 vs 평점 산점도
axes[2].scatter(df["log_review_count"], df["rating"], alpha=0.25, s=12, color="#5B9BD5")
# 추세선
z = np.polyfit(df["log_review_count"], df["rating"], 1)
p_line = np.poly1d(z)
x_line = np.linspace(df["log_review_count"].min(), df["log_review_count"].max(), 100)
r_val = df["log_review_count"].corr(df["rating"])
axes[2].plot(x_line, p_line(x_line), color="red", linewidth=2, label=f"r = {r_val:.3f}")
axes[2].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[2].set_ylabel("평점", fontsize=11)
axes[2].set_title("(c) 리뷰 수 vs 평점", fontsize=12)
axes[2].legend(fontsize=10)

plt.suptitle("Figure 4. 주요 독립변수와 평점의 관계", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %% 3.5 행정구별 평점 분포
# 평균 평점 순으로 정렬
district_order = df.groupby("district")["rating"].mean().sort_values(ascending=False).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) 박스플롯
sns.boxplot(data=df, x="district", y="rating", order=district_order,
            palette="coolwarm_r", ax=axes[0], width=0.6)
axes[0].axhline(df["rating"].mean(), color="red", linestyle="--", alpha=0.5, 
                label=f'전체 평균: {df["rating"].mean():.2f}')
axes[0].set_xlabel("행정구", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 행정구별 평점 분포", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(fontsize=9)

# (b) 평균 + 95% 신뢰구간
dist_stats = df.groupby("district")["rating"].agg(["mean", "std", "count"])
dist_stats["se"] = dist_stats["std"] / np.sqrt(dist_stats["count"])
dist_stats["ci95"] = dist_stats["se"] * 1.96
dist_stats = dist_stats.sort_values("mean", ascending=True)

axes[1].barh(range(len(dist_stats)), dist_stats["mean"],
             xerr=dist_stats["ci95"], color="#5B9BD5", edgecolor="white",
             capsize=3, alpha=0.8)
axes[1].set_yticks(range(len(dist_stats)))
axes[1].set_yticklabels(dist_stats.index)
axes[1].axvline(df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                label=f'전체 평균: {df["rating"].mean():.2f}')
axes[1].set_xlabel("평균 평점 (95% 신뢰구간)", fontsize=11)
axes[1].set_title("(b) 행정구별 평균 평점과 95% 신뢰구간", fontsize=12)
axes[1].legend(fontsize=9)

plt.suptitle("Figure 5. 행정구별 평점 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 수치 요약
print("행정구별 요약 (평균 평점 순):")
summary = df.groupby("district")["rating"].agg(["count", "mean", "median", "std"]).round(3)
print(summary.sort_values("mean", ascending=False).to_string())
# %% 3.6 업종 × 가격대 교차 분석
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 업종 × 가격대별 평점
sns.boxplot(data=df, x="price_category", y="rating", hue="business_type",
            order=["저가", "중가", "고가"], palette=["#5B9BD5", "#ED7D31"],
            ax=axes[0], width=0.6)
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 업종 × 가격대별 평점", fontsize=12)
axes[0].legend(title="업종", fontsize=10)

# (b) 업종 × 리뷰구간별 평점
review_order = ["~50", "51~100", "101~300", "301~1000", "1000+"]
sns.pointplot(data=df, x="review_group", y="rating", hue="business_type",
              order=review_order, palette=["#5B9BD5", "#ED7D31"],
              ax=axes[1], dodge=True, markers=["o", "s"], linestyles=["-", "--"],
              errorbar="ci", capsize=0.1)
axes[1].set_xlabel("리뷰 수 구간", fontsize=11)
axes[1].set_ylabel("평균 평점", fontsize=11)
axes[1].set_title("(b) 업종 × 리뷰 수 구간별 평균 평점", fontsize=12)
axes[1].legend(title="업종", fontsize=10)

plt.suptitle("Figure 6. 업종과 다른 변수의 교호작용 탐색", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %% 3.7 기술통계 요약표
# 전체 기술통계
desc = df[["rating", "review_count", "price_level"]].describe().round(3)
desc.index = ["관측치 수", "평균", "표준편차", "최솟값", "Q1(25%)", "중앙값(50%)", "Q3(75%)", "최댓값"]
desc.columns = ["평점(rating)", "리뷰 수(review_count)", "가격대(price_level)"]
print("Table 1. 주요 변수 기술통계량")
print(desc.to_string())

# %% Part2 소주제별 분석
# 소주제 1
from scipy.stats import kruskal, mannwhitneyu, shapiro, levene, f_oneway
# 가격대별 그룹 분리
price_groups = {cat: df[df["price_category"] == cat]["rating"] for cat in ["저가", "중가", "고가"]}

# 기술통계 테이블
desc_table = pd.DataFrame({
    cat: {
        "n": len(g),
        "평균": f"{g.mean():.3f}",
        "중앙값": f"{g.median():.1f}",
        "표준편차": f"{g.std():.3f}",
        "최솟값": f"{g.min():.1f}",
        "최댓값": f"{g.max():.1f}",
        "Q1": f"{g.quantile(0.25):.1f}",
        "Q3": f"{g.quantile(0.75):.1f}",
    } for cat, g in price_groups.items()
})
 
print("Table 2. 가격대별 평점 기술통계량")
print(desc_table.to_string())

#%% 가격대별 평점 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
order = ["저가", "중가", "고가"]
colors = ["#70AD47", "#5B9BD5", "#ED7D31"]
 
# (a) 박스플롯
sns.boxplot(data=df, x="price_category", y="rating", order=order,
            palette=colors, ax=axes[0], width=0.5)
# 평균 마커
for i, cat in enumerate(order):
    mean_val = price_groups[cat].mean()
    axes[0].scatter(i, mean_val, color="red", s=80, zorder=5, marker="D", edgecolors="white", linewidth=1)
    axes[0].text(i + 0.2, mean_val, f"{mean_val:.3f}", color="red", fontsize=10, fontweight="bold")
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 박스플롯", fontsize=12)
 
# (b) 바이올린 플롯
sns.violinplot(data=df, x="price_category", y="rating", order=order,
               palette=colors, ax=axes[1], inner="quartile")
axes[1].set_xlabel("가격대", fontsize=11)
axes[1].set_ylabel("평점", fontsize=11)
axes[1].set_title("(b) 바이올린 플롯", fontsize=12)
 
# (c) 히스토그램 (겹쳐서)
for cat, color in zip(order, colors):
    axes[2].hist(price_groups[cat], bins=20, alpha=0.5, color=color, label=cat, density=True, edgecolor="white")
axes[2].set_xlabel("평점", fontsize=11)
axes[2].set_ylabel("밀도", fontsize=11)
axes[2].set_title("(c) 가격대별 평점 밀도", fontsize=12)
axes[2].legend()
 
plt.suptitle("Figure 7. 소주제 1 — 가격대별 평점 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%%
# ── (1) 정규성 검정: Shapiro-Wilk ──
print("(1) 정규성 검정 (Shapiro-Wilk)")
print(f"    H_0: 해당 그룹의 평점은 정규분포를 따른다")
print(f"    alpha = 0.05\n")
 
normality_results = {}
for cat in order:
    g = price_groups[cat]
    # 표본이 5000개 초과하면 샘플링
    sample = g.sample(min(500, len(g)), random_state=42) if len(g) > 500 else g
    stat, p = shapiro(sample)
    is_normal = p >= 0.05
    normality_results[cat] = is_normal
    print(f"    {cat} (n={len(g)}): W = {stat:.4f}, p = {p:.6f} → {'정규분포 check' if is_normal else '비정규 x'}")
 
print(f"\n    결론: 저가, 중가가 비정규 -> ANOVA 가정 불충족")
 
# ── (2) 등분산성 검정: Levene ──
lev_stat, lev_p = levene(*price_groups.values())
print(f"\n(2) 등분산성 검정 (Levene)")
print(f"    H_0: 세 그룹의 분산이 동일하다")
print(f"    통계량 = {lev_stat:.4f}, p = {lev_p:.6f}")
print(f"    -> {'등분산 check!' if lev_p >= 0.05 else '이분산(등분산 가정 불충족)'}")
 
# 시각화: 정규성 Q-Q plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (cat, color) in enumerate(zip(order, colors)):
    from scipy.stats import probplot
    probplot(price_groups[cat], dist="norm", plot=axes[i])
    axes[i].set_title(f"{cat} Q-Q Plot (n={len(price_groups[cat])})", fontsize=11)
    axes[i].get_lines()[0].set_color(color)
plt.suptitle("Figure 8. 가격대별 평점 정규성 확인 (Q-Q Plot)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
 
print(f"\n 정규성과 등분산성 모두 충족되지 않으므로,")
print(f"   비모수 검정인 Kruskal-Wallis 검정을 주 검정으로 사용한다.")
print(f"   ANOVA는 참고용으로 함께 보고한다.")

#%%
# ── Kruskal-Wallis 검정 ──
kw_stat, kw_p = kruskal(*price_groups.values())
 
print("Kruskal-Wallis 검정")
print(f"  H_0: 세 가격대의 평점 분포가 동일하다")
print(f"  H_1: 적어도 하나의 가격대에서 평점 분포가 다르다")
print(f"  alpha = 0.05")
print(f"")
print(f"  검정통계량 H = {kw_stat:.4f}")
print(f"  p-value     = {kw_p:.6f}")
print(f"  결론: p = {kw_p:.4f} {'< 0.05 -> H₀ 기각. 가격대별 평점에 유의한 차이가 있다.' if kw_p < 0.05 else '>= 0.05 -> H_0 기각 불가.'}")
 
# ── 효과 크기 (eta^2 근사) ──
# eta^2 = (H - k + 1) / (N - k), k=그룹 수
n_total = len(df)
k = 3
eta_sq = (kw_stat - k + 1) / (n_total - k)
print(f"\n  효과 크기 (eta^2 ≈ {eta_sq:.4f})")
print(f"  -> {'작은 효과 (eta^2 < 0.06)' if eta_sq < 0.06 else ('중간 효과' if eta_sq < 0.14 else '큰 효과')}")
print(f"  -> 통계적으로 유의하지만 실질적 차이는 크지 않음")
 
# ── ANOVA (참고) ──
f_stat, f_p = f_oneway(*price_groups.values())
print(f"\n[참고] One-way ANOVA")
print(f"  F = {f_stat:.4f}, p = {f_p:.6f}")
print(f"  -> {'유의함' if f_p < 0.05 else '유의하지 않음'} (정규성·등분산 가정 미충족이므로 참고용)")

#%% 사후검정
print("사후검정: Mann-Whitney U (Bonferroni 보정)")
print(f"  비교 횟수: 3회 → Bonferroni 보정 적용 (alpha' = 0.05 / 3 = 0.0167)")
print()
 
pairs = [("저가", "중가"), ("저가", "고가"), ("중가", "고가")]
posthoc_results = []
 
for g1, g2 in pairs:
    stat, p = mannwhitneyu(price_groups[g1], price_groups[g2], alternative="two-sided")
    p_adj = min(p * len(pairs), 1.0)  # Bonferroni 보정
    
    # 효과 크기 (rank-biserial correlation)
    n1, n2 = len(price_groups[g1]), len(price_groups[g2])
    r_effect = 1 - (2 * stat) / (n1 * n2)
    
    sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "n.s."))
    
    posthoc_results.append({
        "비교": f"{g1} vs {g2}",
        "U": f"{stat:.0f}",
        "p (원본)": f"{p:.6f}",
        "p (보정)": f"{p_adj:.6f}",
        "r (효과크기)": f"{r_effect:.4f}",
        "판정": sig,
    })
    
    print(f"  {g1} vs {g2}:")
    print(f"    U = {stat:.0f}, p = {p:.6f}, p_adj = {p_adj:.6f} {sig}")
    print(f"    효과 크기 r = {r_effect:.4f} ({'작은' if abs(r_effect) < 0.3 else ('중간' if abs(r_effect) < 0.5 else '큰')} 효과)")
    print()
 
posthoc_df = pd.DataFrame(posthoc_results)
print("\nTable 3. 사후검정 결과 요약")
print(posthoc_df.to_string(index=False))

#%% 소주제2
# 리뷰 수와 평점의 상관분석
from scipy.stats import pearsonr, spearmanr, levene, kruskal, mannwhitneyu

# ── Pearson / Spearman 상관계수 ──
r_pearson, p_pearson = pearsonr(df["review_count"], df["rating"])
r_spearman, p_spearman = spearmanr(df["review_count"], df["rating"])
r_log_pearson, p_log_pearson = pearsonr(df["log_review_count"], df["rating"])
 
print("상관분석 결과")
print(f"{'─'*60}")
print(f"{'방법':<25} {'상관계수':<12} {'p-value':<15} {'판정'}")
print(f"{'─'*60}")
print(f"{'Pearson (원본)':<25} {r_pearson:<12.4f} {p_pearson:<15.6f} {'*' if p_pearson < 0.05 else 'n.s.'}")
print(f"{'Spearman (순위)':<25} {r_spearman:<12.4f} {p_spearman:<15.6f} {'***' if p_spearman < 0.001 else '*' if p_spearman < 0.05 else 'n.s.'}")
print(f"{'Pearson (log 변환)':<25} {r_log_pearson:<12.4f} {p_log_pearson:<15.6f} {'*' if p_log_pearson < 0.05 else 'n.s.'}")
print(f"{'─'*60}")
print(f"\n해석:")
print(f"  - 세 방법 모두 약한 음의 상관이 유의하게 나타남 (p < 0.05)")
print(f"  - Spearman rou = {r_spearman:.4f}로 가장 강한 관계 -> 비선형 단조 관계 존재")
print(f"  - 방향: 리뷰 수가 많을수록 평점이 약간 낮아지는 경향")

#%% 산점도 시각화

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# (a) 원본 스케일
axes[0].scatter(df["review_count"], df["rating"], alpha=0.2, s=10, color="#5B9BD5")
axes[0].set_xlabel("리뷰 수", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title(f"(a) 리뷰 수 vs 평점 (원본)\nPearson r = {r_pearson:.4f}", fontsize=12)
axes[0].set_xlim(0, df["review_count"].quantile(0.98))
 
# (b) 로그 스케일 + 추세선
axes[1].scatter(df["log_review_count"], df["rating"], alpha=0.2, s=10, color="#5B9BD5")
# 추세선
z = np.polyfit(df["log_review_count"], df["rating"], 1)
p_line = np.poly1d(z)
x_line = np.linspace(df["log_review_count"].min(), df["log_review_count"].max(), 100)
axes[1].plot(x_line, p_line(x_line), color="red", linewidth=2, 
             label=f"추세선 (기울기={z[0]:.4f})")
axes[1].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
axes[1].set_ylabel("평점", fontsize=11)
axes[1].set_title(f"(b) log 리뷰 수 vs 평점\nSpearman rou = {r_spearman:.4f} (p < 0.001)", fontsize=12)
axes[1].legend(fontsize=10)
 
plt.suptitle("Figure 10. 소주제 2 — 리뷰 수와 평점의 관계", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 리뷰 수 구간별 평점 분석
# 구간별 기술통계
review_order = ["~50", "51~100", "101~300", "301~1000", "1000+"]
 
grp_stats = df.groupby("review_group", observed=True)["rating"].agg(
    ["count", "mean", "median", "std"]
).reindex(review_order).round(3)
grp_stats.columns = ["n", "평균", "중앙값", "표준편차"]
 
print("Table 4. 리뷰 수 구간별 평점 기술통계량")
print(grp_stats.to_string())
 
print(f"\n핵심 관찰:")
print(f"  평균: {grp_stats['평균'].iloc[0]:.3f} (~50건) → {grp_stats['평균'].iloc[-1]:.3f} (1000+건)")
print(f"  표준편차: {grp_stats['표준편차'].iloc[0]:.3f} (~50건) → {grp_stats['표준편차'].iloc[-1]:.3f} (1000+건)")
print(f"  -> 리뷰 수 증가에 따라 평점이 소폭 하락하고, 표준편차가 크게 감소")

#%% 구간별 시각화 평점 수준과 분산
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
 
# (a) 구간별 박스플롯
sns.boxplot(data=df, x="review_group", y="rating", order=review_order,
            palette="viridis", ax=axes[0], width=0.6)
axes[0].set_xlabel("리뷰 수 구간", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 구간별 평점 분포", fontsize=12)
 
# (b) 구간별 평균 + 95% CI
means = grp_stats["평균"]
stds = grp_stats["표준편차"]
ns = grp_stats["n"]
sems = stds / np.sqrt(ns)
ci95 = sems * 1.96
 
axes[1].errorbar(range(len(review_order)), means, yerr=ci95, 
                 fmt="o-", color="#5B9BD5", linewidth=2, markersize=8, capsize=5)
axes[1].set_xticks(range(len(review_order)))
axes[1].set_xticklabels(review_order)
axes[1].set_xlabel("리뷰 수 구간", fontsize=11)
axes[1].set_ylabel("평균 평점 (95% CI)", fontsize=11)
axes[1].set_title("(b) 구간별 평균 평점 추이", fontsize=12)
 
# (c) 구간별 표준편차 (분산 비교) — 핵심!
colors_var = plt.cm.Reds(np.linspace(0.3, 0.8, len(review_order)))
bars = axes[2].bar(review_order, stds, color=colors_var, edgecolor="white")
for bar, val in zip(bars, stds):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}", 
                 ha="center", fontsize=10, fontweight="bold")
axes[2].set_xlabel("리뷰 수 구간", fontsize=11)
axes[2].set_ylabel("평점 표준편차", fontsize=11)
axes[2].set_title("(c) 구간별 평점 표준편차\n→ 리뷰 많을수록 분산 감소", fontsize=12)
 
plt.suptitle("Figure 11. 소주제 2 — 리뷰 수 구간별 평점 수준과 분산", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 통계검정
# ── (1) 구간별 평점 수준 차이: Kruskal-Wallis ──
groups_data = [df[df["review_group"] == g]["rating"].dropna() for g in review_order]
 
kw_stat, kw_p = kruskal(*groups_data)
print("(1) 구간별 평점 수준 차이 (Kruskal-Wallis)")
print(f"  H_0: 리뷰 수 구간에 따라 평점 분포에 차이가 없다")
print(f"  H = {kw_stat:.4f}, p = {kw_p:.6f}")
print(f"  -> {'유의한 차이 있음' if kw_p < 0.05 else '유의하지 않음'} (alpha = 0.05)")
 
# 효과 크기
n_total = len(df)
eta_sq = (kw_stat - len(review_order) + 1) / (n_total - len(review_order))
print(f"  효과 크기 eta^2 ≈ {eta_sq:.4f} ({'작은' if eta_sq < 0.06 else '중간'} 효과)")
 
# ── (2) 구간별 분산 차이: Levene 검정 ──
lev_stat, lev_p = levene(*groups_data)
print(f"\n(2) 구간별 평점 분산 차이 (Levene)")
print(f"  H_0: 리뷰 수 구간에 따라 평점의 분산이 동일하다")
print(f"  F = {lev_stat:.4f}, p = {lev_p:.6f}")
print(f"  -> {'분산에 유의한 차이 있음' if lev_p < 0.05 else '유의하지 않음'} (alpha = 0.05)")
 
# ── (3) 사후 비교: 인접 구간끼리 ──
print(f"\n(3) 인접 구간 사후비교 (Mann-Whitney U)")
for i in range(len(review_order) - 1):
    g1_name = review_order[i]
    g2_name = review_order[i + 1]
    g1 = df[df["review_group"] == g1_name]["rating"]
    g2 = df[df["review_group"] == g2_name]["rating"]
    stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"  {g1_name} vs {g2_name}: U={stat:.0f}, p={p:.4f} {sig}  (평균: {g1.mean():.3f} vs {g2.mean():.3f})")
    
#%% 대수의 법칙 관점 시각화
fig, ax = plt.subplots(figsize=(10, 6))
 
# 리뷰 수별 평점 산점도 (로그 스케일)
scatter = ax.scatter(df["log_review_count"], df["rating"], 
                     alpha=0.15, s=10, color="#5B9BD5", label="개별 가게")
 
# 구간별 평균 + 표준편차 범위
midpoints = []
for g in review_order:
    sub = df[df["review_group"] == g]
    mid_log = sub["log_review_count"].median()
    midpoints.append(mid_log)
 
grp_means = [df[df["review_group"] == g]["rating"].mean() for g in review_order]
grp_stds = [df[df["review_group"] == g]["rating"].std() for g in review_order]
 
ax.errorbar(midpoints, grp_means, yerr=grp_stds, fmt="D-", color="red", 
            linewidth=2.5, markersize=10, capsize=6, capthick=2,
            label="구간 평균 ± 1 SD", zorder=5)
 
# 전체 평균선
ax.axhline(df["rating"].mean(), color="gray", linestyle=":", alpha=0.7,
           label=f'전체 평균 ({df["rating"].mean():.2f})')
 
ax.set_xlabel("log(1 + 리뷰 수)", fontsize=12)
ax.set_ylabel("평점", fontsize=12)
ax.set_title("Figure 12. 리뷰 수 증가에 따른 평점의 수렴 (대수의 법칙)\n"
             "-> 리뷰가 쌓일수록 평점이 전체 평균 근처로 수렴하고 변동성이 감소",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
 
plt.tight_layout()
plt.show()

#%% 소주제3
from scipy.stats import kruskal, mannwhitneyu, shapiro, levene, f_oneway

# 행정구별 기술통계 (평균 순 정렬)
district_stats = df.groupby("district")["rating"].agg(
    ["count", "mean", "median", "std"]
).round(3).sort_values("mean", ascending=False)
district_stats.columns = ["n", "평균", "중앙값", "표준편차"]
 
print("Table 5. 행정구별 평점 기술통계량 (평균 내림차순)")
print(district_stats.to_string())
 
print(f"\n전체 평균: {df['rating'].mean():.3f}")
print(f"행정구 간 평균의 범위: {district_stats['평균'].min():.3f} ~ {district_stats['평균'].max():.3f}")
print(f"행정구 간 평균의 차이: {district_stats['평균'].max() - district_stats['평균'].min():.3f}")

#%% 행정구별 평점 분포 시각화
district_order = district_stats.index.tolist()  # 평균 순
 
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
 
# (a) 박스플롯
sns.boxplot(data=df, x="district", y="rating", order=district_order,
            palette="coolwarm_r", ax=axes[0], width=0.6)
axes[0].axhline(df["rating"].mean(), color="red", linestyle="--", alpha=0.6,
                label=f'전체 평균: {df["rating"].mean():.3f}')
# 각 구 평균 마커
for i, d in enumerate(district_order):
    mean_val = df[df["district"] == d]["rating"].mean()
    axes[0].scatter(i, mean_val, color="red", s=40, zorder=5, marker="D")
axes[0].set_xlabel("")
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 행정구별 평점 분포 (박스플롯, 평균 순)", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(fontsize=10)
 
# (b) 평균 + 95% 신뢰구간
dist_ci = district_stats.copy()
dist_ci["se"] = dist_ci["표준편차"] / np.sqrt(dist_ci["n"])
dist_ci["ci95"] = dist_ci["se"] * 1.96
dist_ci_sorted = dist_ci.sort_values("평균", ascending=True)
 
colors_bar = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(dist_ci_sorted)))
axes[1].barh(range(len(dist_ci_sorted)), dist_ci_sorted["평균"],
             xerr=dist_ci_sorted["ci95"], color=colors_bar, edgecolor="white",
             capsize=4, alpha=0.85)
axes[1].set_yticks(range(len(dist_ci_sorted)))
axes[1].set_yticklabels([f"{d} (n={int(dist_ci_sorted.loc[d, 'n'])})" for d in dist_ci_sorted.index])
axes[1].axvline(df["rating"].mean(), color="red", linestyle="--", alpha=0.6,
                label=f'전체 평균: {df["rating"].mean():.3f}')
axes[1].set_xlabel("평균 평점", fontsize=11)
axes[1].set_title("(b) 행정구별 평균 평점과 95% 신뢰구간", fontsize=12)
axes[1].legend(fontsize=10)
 
plt.suptitle("Figure 13. 소주제 3 — 행정구별 평점 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 검정 가정 확인
# ── 정규성 ──
print("정규성 검정 (Shapiro-Wilk, 각 행정구)")
non_normal = 0
for d in district_order:
    g = df[df["district"] == d]["rating"]
    sample = g.sample(min(500, len(g)), random_state=42) if len(g) > 500 else g
    stat, p = shapiro(sample)
    is_normal = p >= 0.05
    if not is_normal:
        non_normal += 1
    print(f"  {d:<6s} (n={len(g):>3d}): W={stat:.4f}, p={p:.4f} -> {'정규' if is_normal else '비정규'}")
 
print(f"\n  비정규 분포 행정구: {non_normal}/16개")
print(f"  -> 과반수가 비정규이므로 비모수 검정(Kruskal-Wallis) 사용")
 
# ── 등분산성 ──
district_groups = [df[df["district"] == d]["rating"].values for d in district_order]
lev_stat, lev_p = levene(*district_groups)
print(f"\n등분산성 검정 (Levene)")
print(f"  F = {lev_stat:.4f}, p = {lev_p:.6f}")
print(f"  -> {'등분산' if lev_p >= 0.05 else '이분산'}")

#%% 통계검정

# ── Kruskal-Wallis 검정 ──
kw_stat, kw_p = kruskal(*district_groups)
 
print("Kruskal-Wallis 검정")
print(f"  H_0: 16개 행정구의 평점 분포가 모두 동일하다")
print(f"  H_1: 적어도 하나의 행정구에서 평점 분포가 다르다")
print(f"  alpha = 0.05")
print(f"")
print(f"  검정통계량 H = {kw_stat:.4f}")
print(f"  p-value     = {kw_p:.6f}")
 
if kw_p < 0.05:
    print(f"  결론: p = {kw_p:.4f} < 0.05 → H_1 기각. 행정구별 평점에 유의한 차이가 있다.")
else:
    print(f"  결론: p = {kw_p:.4f} ≥ 0.05 → H_0 기각 불가.")
    print(f"         16개 행정구 전체를 동시에 비교했을 때,")
    print(f"         통계적으로 유의한 차이를 발견할 수 없다.")
 
# 효과 크기
n_total = len(df)
k = 16
eta_sq = (kw_stat - k + 1) / (n_total - k)
print(f"\n  효과 크기 eta² ≈ {eta_sq:.4f} ({'작은 효과' if eta_sq < 0.06 else '중간 효과'})")
 
# ── ANOVA (참고) ──
f_stat, f_p = f_oneway(*district_groups)
print(f"\n[참고] One-way ANOVA: F = {f_stat:.4f}, p = {f_p:.6f}")

#%% 추가 분석: 상위 행정구 vs 하위 행정구 
# 평균 평점 기준 상위 5개 / 하위 5개
top5 = district_stats.head(5).index.tolist()
bot5 = district_stats.tail(5).index.tolist()
 
top5_ratings = df[df["district"].isin(top5)]["rating"]
bot5_ratings = df[df["district"].isin(bot5)]["rating"]
 
stat_tb, p_tb = mannwhitneyu(top5_ratings, bot5_ratings, alternative="two-sided")
 
# 효과 크기 (rank-biserial)
n1, n2 = len(top5_ratings), len(bot5_ratings)
r_effect = 1 - (2 * stat_tb) / (n1 * n2)
 
print("추가 분석: 상위 5개구 vs 하위 5개구")
print(f"  상위 5개구: {', '.join(top5)}")
print(f"    n = {len(top5_ratings)}, 평균 = {top5_ratings.mean():.3f}, 중앙값 = {top5_ratings.median()}")
print(f"  하위 5개구: {', '.join(bot5)}")
print(f"    n = {len(bot5_ratings)}, 평균 = {bot5_ratings.mean():.3f}, 중앙값 = {bot5_ratings.median()}")
print(f"")
print(f"  Mann-Whitney U = {stat_tb:.0f}")
print(f"  p-value = {p_tb:.6f}")
print(f"  효과 크기 r = {r_effect:.4f}")
print(f"  → {'유의한 차이 있음' if p_tb < 0.05 else '유의하지 않음'}")
 
if p_tb < 0.05:
    print(f"\n  해석:")
    print(f"    16개 행정구 전체 비교에서는 유의하지 않았으나,")
    print(f"    평균 평점 상위 5개구와 하위 5개구를 비교하면 유의한 차이가 나타난다.")
    print(f"    상위 5개구의 평균({top5_ratings.mean():.3f})이 하위 5개구({bot5_ratings.mean():.3f})보다")
    print(f"    {top5_ratings.mean() - bot5_ratings.mean():.3f}점 높다.")
    
    
#%% 추가 분석 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
 
# (a) 상위 5 vs 하위 5 비교
df_topbot = df.copy()
df_topbot["group"] = df_topbot["district"].apply(
    lambda x: "상위 5개구" if x in top5 else ("하위 5개구" if x in bot5 else "중간")
)
df_topbot_filtered = df_topbot[df_topbot["group"] != "중간"]
 
sns.boxplot(data=df_topbot_filtered, x="group", y="rating", 
            order=["상위 5개구", "하위 5개구"],
            palette=["#5B9BD5", "#ED7D31"], ax=axes[0], width=0.4)
 
for i, grp in enumerate(["상위 5개구", "하위 5개구"]):
    mean_val = df_topbot_filtered[df_topbot_filtered["group"] == grp]["rating"].mean()
    axes[0].scatter(i, mean_val, color="red", s=80, zorder=5, marker="D", edgecolors="white")
    axes[0].text(i + 0.15, mean_val, f"{mean_val:.3f}", color="red", fontsize=11, fontweight="bold")
 
# 유의성 표시
y_max = 4.6
if p_tb < 0.05:
    axes[0].plot([0, 1], [y_max, y_max], color="black", linewidth=1.5)
    sig_mark = "***" if p_tb < 0.001 else ("**" if p_tb < 0.01 else "*")
    axes[0].text(0.5, y_max + 0.02, sig_mark, ha="center", fontsize=14, fontweight="bold")
 
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title(f"(a) 상위 5개구 vs 하위 5개구\np = {p_tb:.6f}", fontsize=12)
 
# (b) 행정구별 평균 (상위/하위 색 구분)
dist_mean_sorted = district_stats.sort_values("평균", ascending=True)
colors_district = []
for d in dist_mean_sorted.index:
    if d in top5:
        colors_district.append("#5B9BD5")
    elif d in bot5:
        colors_district.append("#ED7D31")
    else:
        colors_district.append("#AAAAAA")
 
axes[1].barh(range(len(dist_mean_sorted)), dist_mean_sorted["평균"],
             color=colors_district, edgecolor="white", alpha=0.85)
axes[1].set_yticks(range(len(dist_mean_sorted)))
axes[1].set_yticklabels(dist_mean_sorted.index)
axes[1].axvline(df["rating"].mean(), color="red", linestyle="--", alpha=0.5,
                label=f'전체 평균: {df["rating"].mean():.3f}')
 
# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#5B9BD5", label="상위 5개구"),
                   Patch(facecolor="#ED7D31", label="하위 5개구"),
                   Patch(facecolor="#AAAAAA", label="중간")]
axes[1].legend(handles=legend_elements, fontsize=9, loc="lower right")
axes[1].set_xlabel("평균 평점", fontsize=11)
axes[1].set_title("(b) 행정구별 평균 평점 (상위/하위 구분)", fontsize=12)
 
plt.suptitle("Figure 14. 소주제 3 — 행정구 그룹별 평점 비교", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 업종별 행정구 패턴

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
for i, bt in enumerate(["음식점", "카페"]):
    sub = df[df["business_type"] == bt]
    d_mean = sub.groupby("district")["rating"].mean().sort_values(ascending=True)
    
    colors_bt = ["#5B9BD5" if d in top5 else ("#ED7D31" if d in bot5 else "#AAAAAA") for d in d_mean.index]
    axes[i].barh(range(len(d_mean)), d_mean.values, color=colors_bt, edgecolor="white", alpha=0.85)
    axes[i].set_yticks(range(len(d_mean)))
    axes[i].set_yticklabels(d_mean.index)
    axes[i].axvline(sub["rating"].mean(), color="red", linestyle="--", alpha=0.5)
    axes[i].set_xlabel("평균 평점", fontsize=11)
    axes[i].set_title(f"{bt} (n={len(sub)})", fontsize=12)
 
plt.suptitle("Figure 15. 업종별 행정구 평균 평점 비교", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 소주제4
# 업종별 기술 통계
from scipy.stats import mannwhitneyu, spearmanr, kruskal, levene
 
rest = df[df["business_type"] == "음식점"]
cafe = df[df["business_type"] == "카페"]
 
# 기술통계 비교
desc_bt = pd.DataFrame({
    "음식점": {
        "n": len(rest),
        "평점 평균": f"{rest['rating'].mean():.3f}",
        "평점 중앙값": f"{rest['rating'].median():.1f}",
        "평점 표준편차": f"{rest['rating'].std():.3f}",
        "리뷰 수 중앙값": f"{rest['review_count'].median():.0f}",
        "리뷰 수 평균": f"{rest['review_count'].mean():.0f}",
    },
    "카페": {
        "n": len(cafe),
        "평점 평균": f"{cafe['rating'].mean():.3f}",
        "평점 중앙값": f"{cafe['rating'].median():.1f}",
        "평점 표준편차": f"{cafe['rating'].std():.3f}",
        "리뷰 수 중앙값": f"{cafe['review_count'].median():.0f}",
        "리뷰 수 평균": f"{cafe['review_count'].mean():.0f}",
    },
})
 
print("Table 6. 업종별 기술통계량 비교")
print(desc_bt.to_string())

#%% 업종별 평점 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors_bt = ["#5B9BD5", "#ED7D31"]
 
# (a) 박스플롯
sns.boxplot(data=df, x="business_type", y="rating", palette=colors_bt, ax=axes[0], width=0.4)
for i, bt in enumerate(["음식점", "카페"]):
    mean_val = df[df["business_type"]==bt]["rating"].mean()
    axes[0].scatter(i, mean_val, color="red", s=80, zorder=5, marker="D", edgecolors="white")
    axes[0].text(i + 0.15, mean_val, f"{mean_val:.3f}", color="red", fontsize=11, fontweight="bold")
axes[0].set_xlabel("업종", fontsize=11)
axes[0].set_ylabel("평점", fontsize=11)
axes[0].set_title("(a) 업종별 평점 분포", fontsize=12)
 
# (b) 겹친 히스토그램
axes[1].hist(rest["rating"], bins=25, alpha=0.6, color=colors_bt[0], label="음식점", density=True, edgecolor="white")
axes[1].hist(cafe["rating"], bins=25, alpha=0.6, color=colors_bt[1], label="카페", density=True, edgecolor="white")
axes[1].axvline(rest["rating"].mean(), color=colors_bt[0], linestyle="--", linewidth=2)
axes[1].axvline(cafe["rating"].mean(), color=colors_bt[1], linestyle="--", linewidth=2)
axes[1].set_xlabel("평점", fontsize=11)
axes[1].set_ylabel("밀도", fontsize=11)
axes[1].set_title("(b) 업종별 평점 밀도 비교", fontsize=12)
axes[1].legend(fontsize=10)
 
# (c) 바이올린 + strip
sns.violinplot(data=df, x="business_type", y="rating", palette=colors_bt, 
               ax=axes[2], inner="quartile", alpha=0.7)
axes[2].set_xlabel("업종", fontsize=11)
axes[2].set_ylabel("평점", fontsize=11)
axes[2].set_title("(c) 바이올린 플롯", fontsize=12)
 
plt.suptitle("Figure 16. 소주제 4 — 업종별 평점 분포", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 업종 간 평점 차이 검정
# Mann-Whitney U 검정
stat, p = mannwhitneyu(rest["rating"], cafe["rating"], alternative="two-sided")
n1, n2 = len(rest), len(cafe)
r_effect = 1 - (2 * stat) / (n1 * n2)
 
print("업종 간 평점 차이 검정 (Mann-Whitney U)")
print(f"  H_0: 음식점과 카페의 평점 분포가 동일하다")
print(f"  H_1: 음식점과 카페의 평점 분포가 다르다")
print(f"  alpha = 0.05")
print(f"")
print(f"  U = {stat:.0f}")
print(f"  p-value = {p:.6f}")
print(f"  효과 크기 r = {r_effect:.4f}")
print(f"")
print(f"  결론: p < 0.001 → H₀ 기각.")
print(f"  카페(평균 {cafe['rating'].mean():.3f})의 평점이")
print(f"  음식점(평균 {rest['rating'].mean():.3f})보다 통계적으로 유의하게 높다.")
print(f"")
print(f"  효과 크기 r = {r_effect:.4f}는 {'작은' if abs(r_effect) < 0.3 else '중간'} 수준이다.")
 
# 분산 비교 (Levene)
lev_stat, lev_p = levene(rest["rating"], cafe["rating"])
print(f"\n분산 비교 (Levene)")
print(f"  F = {lev_stat:.4f}, p = {lev_p:.6f}")
print(f"  -> {'분산에 유의한 차이 있음' if lev_p < 0.05 else '분산 차이 없음'}")
if lev_p < 0.05:
    print(f"  -> 카페(sigma={cafe['rating'].std():.3f})의 평점 분산이")
    print(f"    음식점(sigma={rest['rating'].std():.3f})보다 유의하게 크다.")
    print(f"    이는 카페에 대한 평가가 음식점보다 더 양극화되어 있음을 의미한다.")
    
#%% 업종별 리뷰 수 - 평좀 관계 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
for i, (bt, color, sub) in enumerate(zip(["음식점", "카페"], colors_bt, [rest, cafe])):
    r_sp, p_sp = spearmanr(sub["log_review_count"], sub["rating"])
    
    axes[i].scatter(sub["log_review_count"], sub["rating"], alpha=0.2, s=10, color=color)
    
    # 추세선
    z = np.polyfit(sub["log_review_count"], sub["rating"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(sub["log_review_count"].min(), sub["log_review_count"].max(), 100)
    axes[i].plot(x_line, p_line(x_line), color="red", linewidth=2)
    
    axes[i].set_xlabel("log(1 + 리뷰 수)", fontsize=11)
    axes[i].set_ylabel("평점", fontsize=11)
    sig = "***" if p_sp < 0.001 else ("*" if p_sp < 0.05 else "n.s.")
    axes[i].set_title(f"{bt} (n={len(sub)})\nSpearman ρ = {r_sp:.4f} (p = {p_sp:.4f}) {sig}", fontsize=12)
 
plt.suptitle("Figure 17. 업종별 리뷰 수-평점 관계", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
 
# 수치 비교
print("업종별 리뷰 수-평점 Spearman 상관")
for bt, sub in [("음식점", rest), ("카페", cafe)]:
    r_sp, p_sp = spearmanr(sub["log_review_count"], sub["rating"])
    sig = "***" if p_sp < 0.001 else ("*" if p_sp < 0.05 else "n.s.")
    print(f"  {bt}: ρ = {r_sp:.4f}, p = {p_sp:.6f} {sig}")
 
print(f"\n해석:")
print(f"  음식점: ρ = -0.119 (유의) → 리뷰 많을수록 평점 약간 하락")
print(f"  카페:   ρ = -0.082 (유의하지 않음) → 리뷰 수와 평점 관련 약함")
print(f"  → 음식점에서 리뷰 축적 효과(평점 하락)가 카페보다 더 뚜렷하다.")
print(f"     이는 음식점이 더 다양한 고객층(관광객 포함)으로부터")
print(f"     리뷰를 받아 평점이 '평균으로 회귀'하는 경향이 강하기 때문일 수 있다.")

#%% 업종별 가격대 - 평점 관계 비교(교호작용)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
price_order = ["저가", "중가", "고가"]
 
# (a) 교호작용 플롯 (Point plot)
sns.pointplot(data=df, x="price_category", y="rating", hue="business_type",
              order=price_order, palette=colors_bt, ax=axes[0],
              dodge=0.2, markers=["o", "s"], linestyles=["-", "--"],
              errorbar="ci", capsize=0.1)
axes[0].set_xlabel("가격대", fontsize=11)
axes[0].set_ylabel("평균 평점", fontsize=11)
axes[0].set_title("(a) 업종 × 가격대 교호작용\n→ 선이 평행하지 않으면 교호작용 존재", fontsize=12)
axes[0].legend(title="업종", fontsize=10)
 
# (b) 업종 × 가격대 박스플롯
sns.boxplot(data=df, x="price_category", y="rating", hue="business_type",
            order=price_order, palette=colors_bt, ax=axes[1], width=0.6)
axes[1].set_xlabel("가격대", fontsize=11)
axes[1].set_ylabel("평점", fontsize=11)
axes[1].set_title("(b) 업종 × 가격대별 평점 분포", fontsize=12)
axes[1].legend(title="업종", fontsize=10)
 
plt.suptitle("Figure 18. 소주제 4 — 업종 × 가격대 교호작용", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 교호작용 검정: 부분 F-검정 시행
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression

# ── 변수 생성 ──
df["is_cafe"] = (df["business_type"] == "카페").astype(int)
df["is_low"] = (df["price_category"] == "저가").astype(int)
df["is_mid"] = (df["price_category"] == "중가").astype(int)

# 교호작용항 = 업종 더미 × 가격대 더미
df["cafe_x_low"] = df["is_cafe"] * df["is_low"]
df["cafe_x_mid"] = df["is_cafe"] * df["is_mid"]

y_inter = df["rating"].values

# ── 모델 1: 주효과만 ──
X_main = df[["is_cafe", "is_low", "is_mid"]].values
model_main = LinearRegression().fit(X_main, y_inter)
r2_main = model_main.score(X_main, y_inter)
SS_res_main = np.sum((y_inter - model_main.predict(X_main))**2)

# ── 모델 2: 주효과 + 교호작용 ──
X_inter = df[["is_cafe", "is_low", "is_mid", "cafe_x_low", "cafe_x_mid"]].values
model_inter = LinearRegression().fit(X_inter, y_inter)
r2_inter = model_inter.score(X_inter, y_inter)
SS_res_inter = np.sum((y_inter - model_inter.predict(X_inter))**2)

# ── 부분 F-검정 ──
n = len(y_inter)
p_main = X_main.shape[1]     # 주효과 변수 수 (3)
p_inter = X_inter.shape[1]   # 전체 변수 수 (5)
df_num = p_inter - p_main    # 교호작용 변수 수 (2)
df_den = n - p_inter - 1     # 잔차 자유도

F_stat = ((SS_res_main - SS_res_inter) / df_num) / (SS_res_inter / df_den)
p_value_f = 1 - f_dist.cdf(F_stat, df_num, df_den)

# ── ΔR² (교호작용의 추가 설명력) ──
delta_r2 = r2_inter - r2_main

# 결과 출력
print("교호작용 검정: 부분 F-검정 (Partial F-test)")
print(f"{'─'*55}")
print(f"  모델 1 (주효과만):       R² = {r2_main:.6f}")
print(f"  모델 2 (주효과+교호작용): R² = {r2_inter:.6f}")
print(f"  ΔR² (교호작용 추가 설명력): {delta_r2:.6f} ({delta_r2*100:.4f}%)")
print(f"{'─'*55}")
print(f"  F 통계량 = {F_stat:.4f}")
print(f"  자유도   = ({df_num}, {df_den})")
print(f"  p-value  = {p_value_f:.6f}")
print(f"  결론: {'유의함 (p < 0.05)' if p_value_f < 0.05 else '유의하지 않음 (p ≥ 0.05)'}")

#%% 업종별 리뷰 구간 - 평점 패턴 비교
fig, ax = plt.subplots(figsize=(10, 6))
review_order = ["~50", "51~100", "101~300", "301~1000", "1000+"]
 
sns.pointplot(data=df, x="review_group", y="rating", hue="business_type",
              order=review_order, palette=colors_bt, ax=ax,
              dodge=0.15, markers=["o", "s"], linestyles=["-", "--"],
              errorbar="ci", capsize=0.1)
ax.set_xlabel("리뷰 수 구간", fontsize=11)
ax.set_ylabel("평균 평점 (95% CI)", fontsize=11)
ax.set_title("Figure 19. 업종별 리뷰 수 구간에 따른 평점 추이", fontsize=13, fontweight="bold")
ax.legend(title="업종", fontsize=10)
plt.tight_layout()
plt.show()
 
print("해석:")
print("  두 업종 모두 리뷰 수가 증가하면 평점이 하락하는 추세를 보인다.")
print("  그러나 카페는 모든 리뷰 구간에서 음식점보다 평점이 높으며,")
print("  특히 리뷰가 적은 구간(~50건)에서 카페의 평점이 음식점보다 훨씬 높다.")
print("  리뷰 1000건 이상에서는 두 업종의 평점이 수렴하는 경향이 있다.")
print("")
print("  이는 소규모(리뷰 적은) 카페가 특히 높은 평점을 받는 경향이 있음을 시사하며,")
print("  '숨은 카페' 효과 — 단골 중심의 소수 리뷰어가 높은 점수를 주는 현상 — 로 해석할 수 있다.")

#%% 소주제 5
# 모델링 준비
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import gammaln

# ── 특성 변수 준비 (price를 범주형으로) ──
X = pd.get_dummies(
    df[["log_review_count", "price_category", "district", "business_type"]],
    columns=["price_category", "district", "business_type"],
    drop_first=True,
)
y = df["rating"]

print(f"독립변수: {X.shape[1]}개")
print(f"  수치형 (1개): log_review_count")
print(f"  더미변수 (18개):")
print(f"    price_category 2개 (기준: 고가)")
print(f"    district 15개 (기준: 강서구)")
print(f"    business_type 1개 (기준: 음식점)")
print(f"종속변수: rating (n = {len(y)}, 평균 = {y.mean():.3f}, σ = {y.std():.3f})")

#%% 다중공선성 진단
def calc_vif(X_df):
    """VIF 수동 계산"""
    vif_data = []
    for col in X_df.columns:
        y_temp = X_df[col]
        X_temp = X_df.drop(columns=[col])
        r2 = LinearRegression().fit(X_temp, y_temp).score(X_temp, y_temp)
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_data.append({"변수": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

vif_result = calc_vif(X)
print("Table 8. 전체 독립변수 다중공선성 진단 (VIF)")
print(vif_result.to_string(index=False))
print(f"\nVIF > 10인 변수: {(vif_result['VIF'] > 10).sum()}개")
print(f"최대 VIF: {vif_result['VIF'].max():.2f}")
print(f"\n판단: 모든 VIF < 10 → 심각한 다중공선성 없음")
print(f"  price_category 더미의 VIF가 ~5인 것은 저가/중가가 상호 배타적이기 때문이며 정상 범위임")

#%% 모델 적합
# Train/Test 분리
idx_train, idx_test = train_test_split(range(len(df)), test_size=0.2, random_state=42)
X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

print(f"학습: {len(idx_train)}개 (80%), 테스트: {len(idx_test)}개 (20%)")

# OLS 다중회귀 적합
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

r2_ols = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols = mean_absolute_error(y_test, y_pred_ols)

print(f"\nOLS 다중회귀 성능")
print(f"{'─'*40}")
print(f"  R² (테스트): {r2_ols:.4f}")
print(f"  RMSE:       {rmse_ols:.4f}")
print(f"  MAE:        {mae_ols:.4f}")
print(f"{'─'*40}")
print(f"\n해석: 가격, 리뷰 수, 지역, 업종이 평점 변동의 약 {r2_ols*100:.1f}%만 설명한다.")

#%% 회귀계수 분석
# 회귀 계수 정리
coef_df = pd.DataFrame({
    "변수": X.columns,
    "계수(β)": ols.coef_,
    "|계수|": np.abs(ols.coef_),
}).sort_values("|계수|", ascending=False)

print(f"절편 (β₀) = {ols.intercept_:.4f}")
print(f"\nTable 9. 다중회귀 계수 (절대값 순)")
print(coef_df.to_string(index=False))
 
# %% [markdown]
# ### 5.5 회귀 계수 시각화 + 해석
 
# %%
top_n = 10
top_coefs = coef_df.head(top_n).sort_values("계수(β)")

fig, ax = plt.subplots(figsize=(10, 6))
colors_coef = ["#C44E52" if v < 0 else "#5B9BD5" for v in top_coefs["계수(β)"]]
ax.barh(range(len(top_coefs)), top_coefs["계수(β)"].values, color=colors_coef, edgecolor="white")
ax.set_yticks(range(len(top_coefs)))
ax.set_yticklabels(top_coefs["변수"].values)
ax.axvline(0, color="black", linewidth=0.8)

for i, v in enumerate(top_coefs["계수(β)"].values):
    offset = 0.003 if v >= 0 else -0.003
    ha = "left" if v >= 0 else "right"
    ax.text(v + offset, i, f"{v:.4f}", va="center", ha=ha, fontsize=9, fontweight="bold")

ax.set_xlabel("회귀 계수 (β)", fontsize=11)
ax.set_title("Figure 20. 다중회귀 계수 (상위 10개, price 범주형)\n파랑 = 평점↑ / 빨강 = 평점↓", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 해석
print("회귀 계수 해석:")

# price_category
pc_coefs = coef_df[coef_df["변수"].str.startswith("price_category")]
print(f"\n  1) 가격대 (기준: 고가)")
for _, row in pc_coefs.iterrows():
    cat = row["변수"].replace("price_category_", "")
    print(f"     {cat}: β = {row['계수(β)']:.4f}")
print(f"     → 저가는 고가 대비 평점이 거의 동일 (+0.005)")
print(f"       중가는 고가 대비 약 0.033점 낮음")
print(f"       가격대 효과는 전반적으로 미미 → 소주제 1 결과와 일치")

# log_review_count
lr_coef = coef_df[coef_df["변수"]=="log_review_count"]["계수(β)"].values[0]
print(f"\n  2) log_review_count (β = {lr_coef:.4f})")
print(f"     → 리뷰 수 증가에 따른 평점 변화가 거의 없음")
print(f"     → 다른 변수를 통제하면 리뷰 수의 독립적 효과는 극히 미미")

# business_type
bt_coef = coef_df[coef_df["변수"].str.contains("business_type")]["계수(β)"].values[0]
print(f"\n  3) business_type_카페 (β = {bt_coef:.4f})")
print(f"     → 다른 변수를 통제해도 카페가 음식점보다 약 {bt_coef:.3f}점 높음")
print(f"     → 소주제 4 결과와 일치")

# district
dist_coefs = coef_df[coef_df["변수"].str.startswith("district_")].sort_values("계수(β)")
print(f"\n  4) 행정구 (기준: 강서구)")
print(f"     가장 낮은 구: {dist_coefs.iloc[0]['변수'].replace('district_','')} (β={dist_coefs.iloc[0]['계수(β)']:.4f})")
print(f"     가장 높은 구: {dist_coefs.iloc[-1]['변수'].replace('district_','')} (β={dist_coefs.iloc[-1]['계수(β)']:.4f})")
print(f"     → 모든 행정구가 기준(강서구) 대비 음(-)의 계수 → 강서구가 평점 가장 높음")
print(f"     → 하지만 계수 범위가 {dist_coefs['계수(β)'].min():.3f}~{dist_coefs['계수(β)'].max():.3f}로 작음")

#%% 잔차진단
residuals = y_test.values - y_pred_ols

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) 잔차 vs 예측값
axes[0,0].scatter(y_pred_ols, residuals, alpha=0.25, s=12, color="#5B9BD5")
axes[0,0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0,0].set_xlabel("예측값", fontsize=11)
axes[0,0].set_ylabel("잔차", fontsize=11)
axes[0,0].set_title("(a) 잔차 vs 예측값\n→ 패턴 없으면 선형성·등분산성 충족", fontsize=11)

# (b) 잔차 히스토그램
axes[0,1].hist(residuals, bins=30, color="#5B9BD5", edgecolor="white", density=True, alpha=0.7)
from scipy.stats import norm
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0,1].plot(x_norm, norm.pdf(x_norm, residuals.mean(), residuals.std()),
               "r-", linewidth=2, label="정규분포")
axes[0,1].set_xlabel("잔차", fontsize=11)
axes[0,1].set_ylabel("밀도", fontsize=11)
axes[0,1].set_title("(b) 잔차 분포\n→ 종형이면 정규성 충족", fontsize=11)
axes[0,1].legend()

# (c) Q-Q Plot
from scipy.stats import probplot, shapiro
probplot(residuals, dist="norm", plot=axes[1,0])
axes[1,0].set_title("(c) Q-Q Plot\n→ 대각선에 가까우면 정규분포", fontsize=11)

# (d) 실제 vs 예측
axes[1,1].scatter(y_test, y_pred_ols, alpha=0.25, s=12, color="#5B9BD5")
lims = [min(y_test.min(), y_pred_ols.min())-0.1, max(y_test.max(), y_pred_ols.max())+0.1]
axes[1,1].plot(lims, lims, "r--", linewidth=1.5, label="y = x (완벽한 예측)")
axes[1,1].set_xlabel("실제 평점", fontsize=11)
axes[1,1].set_ylabel("예측 평점", fontsize=11)
axes[1,1].set_title("(d) 실제 vs 예측\n→ 대각선에 가까울수록 정확", fontsize=11)
axes[1,1].legend(fontsize=9)

plt.suptitle("Figure 21. OLS 다중회귀 잔차 진단", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 잔차 통계
res_shapiro = shapiro(residuals[:500] if len(residuals) > 500 else residuals)

print("잔차 진단 요약")
print(f"{'─'*55}")
print(f"  잔차 평균:     {residuals.mean():.4f}")
print(f"  잔차 표준편차:  {residuals.std():.4f}")
print(f"  잔차 왜도:     {pd.Series(residuals).skew():.4f}")
print(f"  Shapiro-Wilk:  W={res_shapiro[0]:.4f}, p={res_shapiro[1]:.6f}")
print(f"{'─'*55}")
print(f"\n진단 결과:")
print(f"  (a) 선형성·등분산성:")
print(f"      잔차 vs 예측값에서 뚜렷한 패턴 없음 → 크게 위배되지 않음")
print(f"      단, 예측값이 {y_pred_ols.min():.2f}~{y_pred_ols.max():.2f}에 밀집")
print(f"      → 모델이 대부분의 가게에 대해 비슷한 값(~4.1)을 예측하고 있음")
print(f"  (b) 정규성:")
print(f"      잔차의 왜도 = {pd.Series(residuals).skew():.2f}로 좌편향 존재")
print(f"      Shapiro-Wilk p < 0.05 → 엄밀한 정규성은 충족되지 않음")
print(f"      단, n={len(residuals)}로 충분히 크므로 중심극한정리에 의해")
print(f"      회귀 계수의 추론은 근사적으로 유효")
print(f"  (c) Q-Q Plot: 양 끝단 이탈 → 극단적 평점(2~3점대, 5점)에서 예측 부정확")
print(f"  (d) 실제 vs 예측: 예측이 좁은 범위에 집중 → 실제 평점의 다양성 미반영")
#%% 모델 구조 변경을 통한 검증: "모델의 문제인가, 변수의 한계인가"
# ── (1) Beta 회귀 ──
# 평점을 (rating - 1) / 4로 변환하여 (0, 1) 범위로 만듦
# Beta 분포는 (0, 1) 범위의 연속형 데이터에 이론적으로 적합

y_beta_all = ((df["rating"] - 1) / 4).clip(1e-6, 1-1e-6)
yb_train = y_beta_all.iloc[idx_train]
yb_test = y_beta_all.iloc[idx_test]

X_train_np = X_train.values.astype(float)
X_test_np = X_test.values.astype(float)
n_feat = X_train.shape[1]

def logit_fn(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def beta_negll(params, X, y):
    """Beta 분포의 음의 로그우도"""
    beta = params[:X.shape[1]+1]
    phi = np.exp(np.clip(params[X.shape[1]+1], -10, 10))  # precision parameter
    mu = logit_fn(beta[0] + X @ beta[1:])
    a = np.clip(mu * phi, 1e-8, 1e6)
    b = np.clip((1 - mu) * phi, 1e-8, 1e6)
    return -np.sum(gammaln(phi) - gammaln(a) - gammaln(b) + (a-1)*np.log(y) + (b-1)*np.log(1-y))

# OLS 결과를 초기값으로 활용
ols_init = LinearRegression().fit(X_train_np, yb_train.values)
init_b = np.zeros(n_feat + 2)
if 0 < ols_init.intercept_ < 1:
    init_b[0] = np.log(ols_init.intercept_ / (1 - ols_init.intercept_))
init_b[1:n_feat+1] = ols_init.coef_ * 4
init_b[-1] = np.log(30)

res_beta = minimize(beta_negll, init_b, args=(X_train_np, yb_train.values),
                    method="L-BFGS-B", options={"maxiter": 10000})

bp = res_beta.x[:n_feat+1]
y_pred_beta = logit_fn(bp[0] + X_test_np @ bp[1:]) * 4 + 1  # 원래 스케일 복원

r2_beta = r2_score(y_test, y_pred_beta)
rmse_beta = np.sqrt(mean_squared_error(y_test, y_pred_beta))
mae_beta = mean_absolute_error(y_test, y_pred_beta)
print(f"Beta 회귀: R²={r2_beta:.4f}, RMSE={rmse_beta:.4f}, MAE={mae_beta:.4f} (수렴: {res_beta.success})")

# ── (2) Gamma GLM (log link) ──
def gamma_negll(params, X, y):
    """Gamma 분포의 음의 로그우도"""
    beta = params[:X.shape[1]+1]
    alpha = np.exp(np.clip(params[X.shape[1]+1], -10, 10))  # shape parameter
    mu = np.exp(np.clip(beta[0] + X @ beta[1:], -10, 10))
    return -np.sum(alpha*np.log(alpha/mu) + (alpha-1)*np.log(y) - alpha*y/mu - gammaln(alpha))

init_g = np.zeros(n_feat + 2)
init_g[0] = np.log(y_train.mean())
init_g[-1] = np.log(5)
res_gamma = minimize(gamma_negll, init_g, args=(X_train_np, y_train.values.astype(float)),
                     method="L-BFGS-B", options={"maxiter": 10000})

gp = res_gamma.x[:n_feat+1]
y_pred_gamma = np.exp(gp[0] + X_test_np @ gp[1:])

r2_gamma = r2_score(y_test, y_pred_gamma)
rmse_gamma = np.sqrt(mean_squared_error(y_test, y_pred_gamma))
mae_gamma = mean_absolute_error(y_test, y_pred_gamma)
print(f"Gamma GLM: R²={r2_gamma:.4f}, RMSE={rmse_gamma:.4f}, MAE={mae_gamma:.4f} (수렴: {res_gamma.success})")

# ── (3) Multinomial Logit (순서형 접근) ──
# 평점을 6개 범주로 이산화하여 분류 문제로 접근
df["rating_cat"] = pd.cut(df["rating"], bins=[0, 3.5, 3.9, 4.1, 4.3, 4.5, 5.1],
                           labels=["~3.5", "3.6~3.9", "4.0~4.1", "4.2~4.3", "4.4~4.5", "4.6~5.0"])
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)
mlr = LogisticRegression(max_iter=5000, random_state=42)
mlr.fit(X_tr_s, df["rating_cat"].iloc[idx_train])
ord_acc = mlr.score(X_te_s, df["rating_cat"].iloc[idx_test])
baseline_acc = df["rating_cat"].value_counts(normalize=True).max()
print(f"Multinomial Logit: Accuracy={ord_acc:.4f} (Baseline={baseline_acc:.4f})")

#%% 모델 비교 종합
print("Table 10. 모델 성능 종합 비교")
print(f"{'─'*70}")
print(f"{'모델':<25} {'분포 가정':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print(f"{'─'*70}")
print(f"{'OLS 다중회귀':<25} {'정규':<15} {r2_ols:<10.4f} {rmse_ols:<10.4f} {mae_ols:<10.4f}")
print(f"{'Beta 회귀 (logit)':<25} {'Beta':<15} {r2_beta:<10.4f} {rmse_beta:<10.4f} {mae_beta:<10.4f}")
print(f"{'Gamma GLM (log)':<25} {'Gamma':<15} {r2_gamma:<10.4f} {rmse_gamma:<10.4f} {mae_gamma:<10.4f}")
print(f"{'─'*70}")
print(f"{'Multinomial Logit':<25} {'다항':<15} Accuracy = {ord_acc:.4f} (Baseline = {baseline_acc:.4f})")
print(f"{'─'*70}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) R² 비교
model_names = ["OLS\n다중회귀", "Beta\n회귀", "Gamma\nGLM"]
r2_vals = [r2_ols, r2_beta, r2_gamma]
colors_m = ["#4C72B0", "#55A868", "#C44E52"]
bars = axes[0].bar(model_names, r2_vals, color=colors_m, edgecolor="white", width=0.5)
axes[0].axhline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, max(val, 0) + 0.003,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("R²", fontsize=11)
axes[0].set_title("(a) 모델별 R² 비교\n→ 모델을 바꿔도 R² ≈ 0.03", fontsize=12)
axes[0].set_ylim(-0.2, 0.1)

# (b) RMSE 비교
rmse_vals = [rmse_ols, rmse_beta, rmse_gamma]
bars = axes[1].bar(model_names, rmse_vals, color=colors_m, edgecolor="white", width=0.5)
baseline_rmse = y_test.std()
axes[1].axhline(baseline_rmse, color="red", linestyle="--", alpha=0.7,
                label=f"Baseline σ={baseline_rmse:.3f}")
for bar, val in zip(bars, rmse_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("RMSE", fontsize=11)
axes[1].set_title("(b) 모델별 RMSE 비교", fontsize=12)
axes[1].legend(fontsize=9)

plt.suptitle("Figure 22. 모델 구조 변경에 따른 성능 비교", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% 설명력의 한계: 모델이 아니라 변수의 문제
fig, ax = plt.subplots(figsize=(10, 6))

explained = max(r2_ols, 0) * 100
unexplained = 100 - explained

wedges, texts, autotexts = ax.pie(
    [explained, unexplained],
    labels=None,
    autopct="%1.1f%%",
    colors=["#5B9BD5", "#E0E0E0"],
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=14, fontweight="bold"),
)

ax.legend(
    [f"설명 가능 ({explained:.1f}%)\n가격, 리뷰 수, 지역, 업종",
     f"설명 불가 ({unexplained:.1f}%)\n맛, 서비스, 청결도, 분위기 등"],
    loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11,
)

ax.set_title("Figure 23. 평점 변동의 설명력 분해\n→ 외부 관찰 가능 요인으로 평점의 약 3~4%만 설명 가능",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("핵심 결론: 모델의 문제가 아니라 변수의 한계")
print(f"{'─'*60}")
print(f"  OLS (정규 가정):     R² = {r2_ols:.4f}")
print(f"  Beta (bounded 가정): R² = {r2_beta:.4f}")
print(f"  Gamma (양수 가정):   R² = {r2_gamma:.4f}")
print(f"  Multinomial (순서형): Accuracy = {ord_acc:.4f} ≈ Baseline {baseline_acc:.4f}")
print(f"{'─'*60}")
print(f"")
print(f"  네 가지 서로 다른 모델 구조 — 정규 분포, Beta 분포, Gamma 분포,")
print(f"  순서형 분류 — 를 적용하였으나, 모두 유사하게 낮은 성능을 보였다.")
print(f"")
print(f"  이는 낮은 설명력의 원인이 '모델이 잘못 선택되어서'가 아니라")
print(f"  '투입된 변수 자체가 평점을 설명하는 데 본질적으로 한계가 있기 때문'")
print(f"  임을 실증적으로 입증한다.")
print(f"")
print(f"  평점의 96%는 맛, 서비스, 분위기, 청결도 등 실제 방문 경험에 의해")
print(f"  결정되며, 가격·위치·리뷰 수·업종 같은 외부 관찰 가능 정보만으로는")
print(f"  평점을 예측할 수 없다.")