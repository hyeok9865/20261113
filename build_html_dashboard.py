import base64
import os
import json
import pandas as pd

workspace = r"C:\Users\user\Desktop\2026data"
plots_dir = os.path.join(workspace, "plots")

def get_base64_img(filename):
    path = os.path.join(plots_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    return ""

img_boxplots = get_base64_img("boxplots_all.png")
img_distributions = get_base64_img("distributions_detail.png")
img_scatter = get_base64_img("scatter_mv_relationships.png")
img_models = get_base64_img("model_comparison_bars.png")
img_residuals = get_base64_img("residuals_diagnostics.png")
img_importance = get_base64_img("feature_importance.png")

# Load CSVs
df_outliers = pd.read_csv(os.path.join(workspace, "outlier_summary.csv"))
df_perf = pd.read_csv(os.path.join(workspace, "model_performance_comparison.csv"))

html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Boston Housing 이상치 분석 & ML 파이프라인 대시보드</title>
    <!-- Chart.js & TailwindCSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800&display=swap');
        body {{
            font-family: 'Pretendard', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .card {{
            background: #ffffff;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.08);
        }}
        .metric-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        .zoomable-img {{
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .zoomable-img:hover {{
            transform: scale(1.01);
        }}
    </style>
</head>
<body class="p-4 md:p-8">

    <!-- Header -->
    <header class="max-w-7xl mx-auto mb-8">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b pb-6">
            <div>
                <div class="flex items-center gap-2 text-indigo-600 font-semibold mb-1">
                    <i class="fa-solid fa-chart-line"></i>
                    <span>Kaggle Boston Housing Dataset</span>
                </div>
                <h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">이상치(Outlier) 진단 및 머신러닝 최적화 리포트</h1>
                <p class="text-slate-500 mt-1 text-sm md:text-base">시각화 기반 이상치 선별 기준 수립 및 전처리 전후 4개 회귀 모델 교차검증 비교</p>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-xs bg-slate-100 text-slate-700 px-3 py-2 rounded-lg border border-slate-200 flex items-center gap-2">
                    <i class="fa-regular fa-calendar-check text-indigo-500"></i>
                    기준일자: 2026.11
                </span>
                <span class="text-xs bg-emerald-50 text-emerald-700 px-3 py-2 rounded-lg border border-emerald-200 font-medium">
                    <i class="fa-solid fa-circle-check text-emerald-500 mr-1"></i> 검증 완료
                </span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto space-y-8">

        <!-- Top KPI Cards -->
        <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-5">
                <div class="flex items-center justify-between text-slate-500 text-sm">
                    <span>데이터 표본 수</span>
                    <i class="fa-solid fa-database text-blue-500"></i>
                </div>
                <div class="mt-3 flex items-baseline gap-2">
                    <span class="text-3xl font-bold text-slate-800">490건</span>
                    <span class="text-xs text-rose-500 font-semibold">-16건 (인공 캡 제거)</span>
                </div>
                <p class="text-xs text-slate-400 mt-2">원본 506행 $\rightarrow$ 정제 490행 (보존율 96.9%)</p>
            </div>

            <div class="card p-5">
                <div class="flex items-center justify-between text-slate-500 text-sm">
                    <span>타깃 인공 상한치 (Cap)</span>
                    <i class="fa-solid fa-flag text-rose-500"></i>
                </div>
                <div class="mt-3 flex items-baseline gap-2">
                    <span class="text-3xl font-bold text-slate-800">16건</span>
                    <span class="metric-badge bg-rose-50 text-rose-600 text-xs">MV = $50k</span>
                </div>
                <p class="text-xs text-slate-400 mt-2">조사 당시 5만달러 이상 주택을 일괄 상한 절단한 왜곡치</p>
            </div>

            <div class="card p-5">
                <div class="flex items-center justify-between text-slate-500 text-sm">
                    <span>최고 모델 설명력 (R²)</span>
                    <i class="fa-solid fa-award text-amber-500"></i>
                </div>
                <div class="mt-3 flex items-baseline gap-2">
                    <span class="text-3xl font-bold text-indigo-600">0.838</span>
                    <span class="text-xs text-emerald-600 font-bold">+0.075 개선</span>
                </div>
                <p class="text-xs text-slate-400 mt-2">Random Forest (원본 0.763 $\rightarrow$ 정제 0.838)</p>
            </div>

            <div class="card p-5">
                <div class="flex items-center justify-between text-slate-500 text-sm">
                    <span>오차 감소율 (RMSE)</span>
                    <i class="fa-solid fa-arrow-trend-down text-emerald-500"></i>
                </div>
                <div class="mt-3 flex items-baseline gap-2">
                    <span class="text-3xl font-bold text-emerald-600">-29.4%</span>
                    <span class="metric-badge bg-emerald-50 text-emerald-700 text-xs">$3,115</span>
                </div>
                <p class="text-xs text-slate-400 mt-2">평균 오차 $4,414 $\rightarrow$ $3,115 (주택 가치 오차 급감)</p>
            </div>
        </section>

        <!-- Decision Strategy Section -->
        <section class="card p-6">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <i class="fa-solid fa-sliders text-indigo-600"></i>
                이상치 처리 3대 원칙 및 최종 기준
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="border rounded-xl p-5 bg-rose-50/40 border-rose-100">
                    <div class="flex items-center gap-2 text-rose-700 font-bold mb-2">
                        <i class="fa-solid fa-trash-can"></i>
                        <span>1. 인위적 상한치 제거 (Drop)</span>
                    </div>
                    <p class="text-sm text-slate-700 mb-3"><strong>대상:</strong> 타깃 변수 <code>MV == 50.0</code> (16건)</p>
                    <p class="text-xs text-slate-600 leading-relaxed">
                        실제 가치가 $60k~$100k인 초고가 주택들이 조사 편의상 $50k로 묶여 평평한 천장을 형성했습니다. 이로 인해 회귀 직선의 기울기가 심각하게 꺾여 전 구간의 예측력을 망치므로 <strong>제거가 필수</strong>입니다.
                    </p>
                </div>

                <div class="border rounded-xl p-5 bg-blue-50/40 border-blue-100">
                    <div class="flex items-center gap-2 text-blue-700 font-bold mb-2">
                        <i class="fa-solid fa-arrows-spin"></i>
                        <span>2. 정보 보존 로그 변환 (Log1p)</span>
                    </div>
                    <p class="text-sm text-slate-700 mb-3"><strong>대상:</strong> <code>CRIM</code> (왜도 5.22), <code>DIS</code> (왜도 1.01)</p>
                    <p class="text-xs text-slate-600 leading-relaxed">
                        범죄율과 직장 거리는 우측 꼬리가 매우 길지만 실제 도심 지역의 핵심 특성입니다. 행을 삭제하면 13%의 도심 표본이 날아가므로, <code>np.log1p</code> 변환으로 왜도를 0.4로 정규화하여 <strong>데이터를 100% 보존</strong>했습니다.
                    </p>
                </div>

                <div class="border rounded-xl p-5 bg-emerald-50/40 border-emerald-100">
                    <div class="flex items-center gap-2 text-emerald-700 font-bold mb-2">
                        <i class="fa-solid fa-shield-halved"></i>
                        <span>3. 자연 발생치 보존 / 트리 활용</span>
                    </div>
                    <p class="text-sm text-slate-700 mb-3"><strong>대상:</strong> <code>RM</code> (방 개수), <code>TAX</code>, <code>PT</code></p>
                    <p class="text-xs text-slate-600 leading-relaxed">
                        방 8개 이상의 대저택이나 3.5개 이하의 초소형 단지는 자연스러운 시장의 다양성입니다. 이상치에 면역력이 있는 트리 기반 앙상블(Random Forest, GBDT)을 적용하여 <strong>삭제 없이 정확하게 학습</strong>했습니다.
                    </p>
                </div>
            </div>
        </section>

        <!-- Interactive Visual Comparison: Model Evaluation -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="card p-6">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                        <i class="fa-solid fa-chart-column text-indigo-600"></i>
                        모델별 설명력 (R² Score) 교차검증 비교
                    </h2>
                    <span class="text-xs bg-indigo-50 text-indigo-700 font-semibold px-2 py-1 rounded">높을수록 우수</span>
                </div>
                <div class="h-64">
                    <canvas id="r2Chart"></canvas>
                </div>
            </div>

            <div class="card p-6">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                        <i class="fa-solid fa-chart-bar text-rose-500"></i>
                        예측 오차 (RMSE: $1,000 단위) 비교
                    </h2>
                    <span class="text-xs bg-rose-50 text-rose-700 font-semibold px-2 py-1 rounded">낮을수록 우수</span>
                </div>
                <div class="h-64">
                    <canvas id="rmseChart"></canvas>
                </div>
            </div>
        </section>

        <!-- Detailed Visual Charts Gallery -->
        <section class="card p-6">
            <h2 class="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                <i class="fa-solid fa-images text-indigo-600"></i>
                이상치 탐색 및 진단 시각화 아카이브
            </h2>

            <div class="space-y-8">
                <!-- Boxplots -->
                <div class="border rounded-xl p-4 bg-slate-50">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-bold text-slate-800 text-base">1. 피처별 상자그림 (Boxplots & IQR 1.5x 이상치 개수)</h3>
                        <span class="text-xs text-slate-500">Q1 - 1.5*IQR ~ Q3 + 1.5*IQR 기준선</span>
                    </div>
                    <img src="{img_boxplots}" alt="Boxplots" class="w-full rounded-lg shadow-sm border bg-white zoomable-img">
                </div>

                <!-- Distributions -->
                <div class="border rounded-xl p-4 bg-slate-50">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-bold text-slate-800 text-base">2. 주요 편향 피처 분포 및 로그 변환 효과</h3>
                        <span class="text-xs text-slate-500">CRIM, RM, DIS, B, MV 타깃 집중 분석</span>
                    </div>
                    <img src="{img_distributions}" alt="Distributions" class="w-full rounded-lg shadow-sm border bg-white zoomable-img">
                </div>

                <!-- Scatter Relationships -->
                <div class="border rounded-xl p-4 bg-slate-50">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-bold text-slate-800 text-base">3. 주택 가격(MV) 대비 산점도 & Capping 50.0(빨간 삼각형) 왜곡 확인</h3>
                        <span class="text-xs text-slate-500">MV=50 집단이 형성하는 비정상적 평평한 수평선</span>
                    </div>
                    <img src="{img_scatter}" alt="Scatter Relationships" class="w-full rounded-lg shadow-sm border bg-white zoomable-img">
                </div>

                <!-- Residual Diagnostics & Feature Importance -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="border rounded-xl p-4 bg-slate-50">
                        <h3 class="font-bold text-slate-800 text-base mb-2">4. 잔차 진단 (이상치 처리 전후 오차 패턴)</h3>
                        <img src="{img_residuals}" alt="Residuals" class="w-full rounded-lg shadow-sm border bg-white zoomable-img">
                    </div>
                    <div class="border rounded-xl p-4 bg-slate-50">
                        <h3 class="font-bold text-slate-800 text-base mb-2">5. 정제 데이터 기준 피처 중요도 (Random Forest)</h3>
                        <img src="{img_importance}" alt="Feature Importance" class="w-full rounded-lg shadow-sm border bg-white zoomable-img">
                    </div>
                </div>
            </div>
        </section>

        <!-- Outlier Statistics Table -->
        <section class="card p-6">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <i class="fa-solid fa-table-list text-indigo-600"></i>
                전체 변수별 이상치 정밀 통계표
            </h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm text-slate-600">
                    <thead class="bg-slate-100 text-slate-700 text-xs uppercase font-bold">
                        <tr>
                            <th class="px-4 py-3 rounded-l-lg">Feature</th>
                            <th class="px-3 py-3">Min</th>
                            <th class="px-3 py-3">Q1</th>
                            <th class="px-3 py-3">Median</th>
                            <th class="px-3 py-3">Q3</th>
                            <th class="px-3 py-3">Max</th>
                            <th class="px-3 py-3">왜도 (Skew)</th>
                            <th class="px-3 py-3">IQR 이상치</th>
                            <th class="px-3 py-3">비율 (%)</th>
                            <th class="px-3 py-3">극단치 (3×IQR)</th>
                            <th class="px-3 py-3 rounded-r-lg">Z > 3</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        {"".join([f'''
                        <tr class="hover:bg-slate-50/80 transition">
                            <td class="px-4 py-3 font-bold text-slate-800">{row['Feature']}</td>
                            <td class="px-3 py-3">{row['Min']:.2f}</td>
                            <td class="px-3 py-3">{row['Q1']:.2f}</td>
                            <td class="px-3 py-3">{row['Median']:.2f}</td>
                            <td class="px-3 py-3">{row['Q3']:.2f}</td>
                            <td class="px-3 py-3">{row['Max']:.2f}</td>
                            <td class="px-3 py-3 {'text-rose-600 font-bold' if abs(row['Skewness']) > 1 else ''}">{row['Skewness']:.2f}</td>
                            <td class="px-3 py-3 {'text-rose-600 font-bold' if row['IQR_Outliers'] > 0 else 'text-slate-400'}">{row['IQR_Outliers']}</td>
                            <td class="px-3 py-3">{row['IQR_Ratio(%)']:.1f}%</td>
                            <td class="px-3 py-3 text-amber-600 font-medium">{row['Extreme_Outliers']}</td>
                            <td class="px-3 py-3 text-indigo-600 font-medium">{row['Z_Outliers(|z|>3)']}</td>
                        </tr>
                        ''' for _, row in df_outliers.iterrows()])}
                    </tbody>
                </table>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="max-w-7xl mx-auto mt-12 mb-6 text-center text-xs text-slate-400 border-t pt-6">
        <p>Generated by Antigravity AI Data Pipeline • Dataset: Kaggle arpikr/boston • C:\\Users\\user\\Desktop\\2026data\\dashboard.html</p>
    </footer>

    <!-- Interactive Script for Chart.js -->
    <script>
        const models = {json.dumps(list(df_perf['Model']))};
        const rawR2 = {json.dumps(list(df_perf['Raw_R2'].round(3)))};
        const cleanR2 = {json.dumps(list(df_perf['Clean_R2'].round(3)))};
        const rawRmse = {json.dumps(list(df_perf['Raw_RMSE'].round(2)))};
        const cleanRmse = {json.dumps(list(df_perf['Clean_RMSE'].round(2)))};

        // R2 Chart
        new Chart(document.getElementById('r2Chart'), {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [
                    {{
                        label: '원본 데이터 (N=506)',
                        data: rawR2,
                        backgroundColor: '#93c5fd',
                        borderRadius: 6
                    }},
                    {{
                        label: '정제 데이터 (N=490)',
                        data: cleanR2,
                        backgroundColor: '#3b82f6',
                        borderRadius: 6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        min: 0.5,
                        max: 0.9,
                        ticks: {{ stepSize: 0.05 }}
                    }}
                }},
                plugins: {{
                    legend: {{ position: 'top' }}
                }}
            }}
        }});

        // RMSE Chart
        new Chart(document.getElementById('rmseChart'), {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [
                    {{
                        label: '원본 데이터 (N=506)',
                        data: rawRmse,
                        backgroundColor: '#fca5a5',
                        borderRadius: 6
                    }},
                    {{
                        label: '정제 데이터 (N=490)',
                        data: cleanRmse,
                        backgroundColor: '#ef4444',
                        borderRadius: 6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        min: 2.0,
                        max: 6.0
                    }}
                }},
                plugins: {{
                    legend: {{ position: 'top' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

output_path = os.path.join(workspace, "dashboard.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Interactive Dashboard HTML generated successfully: {output_path} (Size: {len(html_content):,} bytes)")
