# app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="3D Cluster 분류", layout="wide")

# --- 1. 제목 및 설명 ---
st.title("3D Cluster 분류 시각화")
st.write("""
새로운 데이터 포인트를 입력하면,  
1) 각 클러스터 센터와의 거리를 계산하여 어떤 클러스터에 속하는지 예측  
2) 3D Scatter 플롯으로 결과를 확인  
할 수 있습니다.
""")

# --- 2. 데이터 로드 ---
@st.cache_data
def load_data():
    base_dir='data/'
    # 로컬 경로(앱과 동일 디렉터리에 있어야 함)
    df = pd.read_csv(base_dir+"data.csv")
    center_df = pd.read_csv(base_dir+"center_points.csv")
    # centers: 첫 번째 컬럼(인덱스 역할)을 제외한 나머지 3개 칼럼 사용
    centers = center_df.values[:, 1:4]
    return df, centers

df, centers = load_data()

# --- 3. 사용자 입력: 새로운 점수 ---
st.sidebar.header("새로운 데이터 포인트 입력")
env_input = st.sidebar.number_input("Environment Score", min_value=0.0, value=400.0, step=1.0)
soc_input = st.sidebar.number_input("Social Score", min_value=0.0, value=300.0, step=1.0)
gov_input = st.sidebar.number_input("Governance Score", min_value=0.0, value=280.0, step=1.0)
new_point = np.array([env_input, soc_input, gov_input])

# --- 4. 클러스터 이름 정의 ---
cluster_names = ['C0', 'C1', 'C2', 'C3', 'C4']  # 필요에 따라 개수 변경

# --- 5. 기존 데이터에서 3D 변수 추출 ---
X = df[['environment_score', 'social_score', 'governance_score']].values

# --- 6. 각 기존 포인트에 대해 라벨(클러스터) 계산 ---
#    (각 포인트와 centers 간의 거리 계산 후 argmin)
#    centers 모양: (K, 3), X 모양: (N, 3)
#    distances_pts shape = (N, K)
distances_pts = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
labels = np.argmin(distances_pts, axis=1)

# --- 7. 새로운 포인트 예측 클러스터 계산 ---
distances_new = np.linalg.norm(centers - new_point, axis=1)
predicted_cluster = np.argmin(distances_new)
predicted_name = cluster_names[predicted_cluster]

# --- 8. 예측 결과 출력 ---
st.subheader("예측 결과")
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown(f"- **예측 클러스터:** `{predicted_name}`")
    st.markdown("- **센터와의 거리:**")
    for i, d in enumerate(distances_new):
        name = cluster_names[i] if i < len(cluster_names) else f"C{i}"
        st.markdown(f"  - {name}: `{d:.2f}`")

# --- 9. Plotly 3D Scatter 그리기 ---
with col2:
    fig = go.Figure()

    # (1) 클러스터 센터 표시
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=cluster_names,
        textposition="top center",
        name="Cluster Centers"
    ))

    # (2) 기존 데이터 점들 표시 (레이블별 색상)
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=labels,
            colorscale='Viridis',
            opacity=0.4
        ),
        name='Clustered Data'
    ))

    # (3) 새로운 데이터 포인트 표시
    fig.add_trace(go.Scatter3d(
        x=[new_point[0]],
        y=[new_point[1]],
        z=[new_point[2]],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        text=["New Point"],
        textposition="top center",
        name="New Data Point"
    ))

    # (4) 센터와 새로운 점을 연결하는 선
    legend_other_added = False
    for i, center in enumerate(centers):
        x_line = [new_point[0], center[0]]
        y_line = [new_point[1], center[1]]
        z_line = [new_point[2], center[2]]
        if i == predicted_cluster:
            # 예측된 센터: 녹색 두꺼운 선
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='green', width=6),
                name='Closest Center'
            ))
        else:
            # 나머지 센터: 검은색 점선, 범례는 한 번만
            if not legend_other_added:
                fig.add_trace(go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode='lines',
                    line=dict(color='black', dash='dash', width=5),
                    name='Other Centers'
                ))
                legend_other_added = True
            else:
                fig.add_trace(go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    showlegend=False
                ))

    # 레이아웃 설정
    fig.update_layout(
        title="3D KMeans Center 분류",
        scene=dict(
            xaxis_title="Environment Score",
            yaxis_title="Social Score",
            zaxis_title="Governance Score"
        ),
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# --- 10. 하단 설명 ---
st.write("""
---
- **KMeans**를 이용해 미리 계산된 센터 값을 사용하며,  
  각 데이터 포인트는 “센터와의 유클리드 거리”를 기반으로 가장 가까운 클러스터에 할당합니다.

- 새로운 점수를 사이드바에 입력하면, 가장 가까운 클러스터를 예측하고 3D 시각화 결과를 바로 확인할 수 있습니다.
""")
