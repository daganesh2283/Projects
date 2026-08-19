import json
import os
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Crypto Community Risk Dashboard",
    page_icon="🔍",
    layout="wide",
)
dir
def _project_outputs_dir() -> str:
    """Return project outputs directory path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


@st.cache_data(show_spinner=False)
def load_community_stats() -> pd.DataFrame:
    """Load community risk table if available."""
    path = os.path.join(_project_outputs_dir(), "community_stats.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dataset_stats() -> Dict[str, object]:
    """Load dataset summary stats if available."""
    path = os.path.join(_project_outputs_dir(), "dataset_stats.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


@st.cache_data(show_spinner=False)
def load_pipeline_metrics() -> Dict[str, object]:
    """Load evaluation metrics saved by the pipeline."""
    path = os.path.join(_project_outputs_dir(), "pipeline_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


@st.cache_data(show_spinner=False)
def load_explanations() -> Dict[int, str]:
    """Load community explanation strings keyed by community ID."""
    path = os.path.join(_project_outputs_dir(), "community_explanations.csv")
    if not os.path.exists(path):
        return {}

    try:
        exp_df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}

    if "community_id" not in exp_df.columns or "explanation" not in exp_df.columns:
        return {}
    return {int(row["community_id"]): str(row["explanation"]) for _, row in exp_df.iterrows()}


def style_risk_rows(row):
    """Color rows by risk label for table readability."""
    color = ""
    if row["risk_label"] == "HIGH":
        color = "background-color: rgba(255, 0, 0, 0.12);"
    elif row["risk_label"] == "MEDIUM":
        color = "background-color: rgba(255, 165, 0, 0.15);"
    elif row["risk_label"] == "LOW":
        color = "background-color: rgba(0, 128, 0, 0.12);"
    return [color] * len(row)


community_stats = load_community_stats()
dataset_stats = load_dataset_stats()
pipeline_metrics = load_pipeline_metrics()
explanations = load_explanations()
outputs_dir = _project_outputs_dir()
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if community_stats.empty:
    st.title("Crypto Community Risk Dashboard")
    st.warning("No data found! Please upload your dataset to populate the dashboard.")
    st.info("You can upload ONE Excel file with 3 sheets ('features', 'edgelist', 'classes') OR THREE separate CSVs ('features', 'edgelist', 'classes').")
    
    uploaded_files = st.file_uploader("Upload Dataset", type=["xlsx", "csv"], accept_multiple_files=True)
    if uploaded_files:
        os.makedirs(data_dir, exist_ok=True)
        
        is_excel = False
        csv_count = 0
        
        for f in uploaded_files:
            if f.name.endswith(".xlsx"):
                with st.spinner("Processing Excel file..."):
                    df_dict = pd.read_excel(f, sheet_name=None)
                    for sheet_name, df in df_dict.items():
                        if "feature" in sheet_name.lower():
                            df.to_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), index=False, header=False)
                        elif "edge" in sheet_name.lower():
                            df.to_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"), index=False, header=True)
                        elif "class" in sheet_name.lower():
                            df.to_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"), index=False, header=True)
                is_excel = True
                
            elif f.name.endswith(".csv"):
                file_path = os.path.join(data_dir, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                csv_count += 1
                
        if is_excel or csv_count >= 3:
            if not st.session_state.get("pipeline_ran", False):
                st.info("Data uploaded successfully! Precomputing network graph pipeline...")
                import subprocess
                subprocess.run(["python", "main.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
                st.session_state["pipeline_ran"] = True
                st.cache_data.clear()
                st.rerun()
            
    st.stop()


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Community Risk Table",
        "Community Inspector",
        "Evaluation Results",
        "Risk Scorecard",
    ],
)

if page == "Overview":
    st.title("Explainable Graph-Based Detection of Illicit Transaction Communities")
    st.write(
        "Community-level illicit activity analytics for cryptocurrency transaction networks using graph features, "
        "community detection, explainability, and risk scoring."
    )

    source_mode = str(dataset_stats.get("source_mode", "unknown")).upper()
    if source_mode == "ELLIPTIC":
        st.success("Data source: Elliptic dataset (live).")
    elif source_mode == "SYNTHETIC":
        st.warning("Data source: synthetic fallback demo (CSV files not found).")
    else:
        st.info("Data source not available yet. Run the pipeline to populate outputs.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", f"{dataset_stats.get('total_nodes', 0):,}")
    col2.metric("Total Edges", f"{dataset_stats.get('total_edges', 0):,}")
    col3.metric("Communities", f"{dataset_stats.get('communities', len(community_stats)):,}")
    col4.metric("High Risk", f"{dataset_stats.get('high_risk', 0):,}")

    class_dist = dataset_stats.get("class_distribution", {})
    if class_dist:
        class_df = pd.DataFrame(
            {"label": list(class_dist.keys()), "count": list(class_dist.values())}
        )
        pie_fig = px.pie(
            class_df,
            values="count",
            names="label",
            title="Node Class Distribution",
            color="label",
            color_discrete_map={"illicit": "red", "licit": "green", "unknown": "gray"},
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("Class distribution will appear after running the pipeline.")

elif page == "Community Risk Table":
    st.title("Community Risk Table")
    if not community_stats.empty:
        risk_filter = st.selectbox("Filter by risk label", ["All", "HIGH", "MEDIUM", "LOW"])
        sort_order = st.selectbox("Sort risk score", ["Descending", "Ascending"])

        table_df = community_stats.copy()
        if risk_filter != "All":
            table_df = table_df[table_df["risk_label"] == risk_filter]

        table_df = table_df.sort_values(
            by="risk_score",
            ascending=(sort_order == "Ascending"),
        )
        st.dataframe(
            table_df.style.apply(style_risk_rows, axis=1),
            use_container_width=True,
            height=600,
        )

elif page == "Community Inspector":
    st.title("Community Inspector")
    if not community_stats.empty:
        selected_community = st.selectbox(
            "Select community_id",
            sorted(community_stats["community_id"].astype(int).tolist()),
        )
        row = community_stats[community_stats["community_id"] == selected_community].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Score", f"{row['risk_score']:.3f}")
        m2.metric("Risk Label", row["risk_label"])
        m3.metric("Total Nodes", f"{int(row['total_nodes'])}")
        m4.metric("Illicit Ratio", f"{row['illicit_ratio']:.2%}")

        if "attack_type" in row:
            st.info(f"**Identified Attack Profile:** {row['attack_type']}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Betweenness", f"{row['avg_betweenness']:.4f}")
        m6.metric("Avg PageRank", f"{row['avg_pagerank']:.6f}")
        m7.metric("Neighbor Illicit", f"{row['avg_neighbor_illicit']:.3f}")
        m8.metric("Internal Density", f"{row['internal_edge_density']:.3f}")

        st.subheader("Explainability")
        explanation = explanations.get(int(selected_community))
        if explanation:
            st.info(explanation)
        else:
            st.info("No explanation found for this community yet.")

        subgraph_path = os.path.join(outputs_dir, "high_risk_subgraph.png")
        if os.path.exists(subgraph_path):
            st.subheader("Subgraph Visualization")
            st.image(Image.open(subgraph_path), caption="Top high-risk community subgraph view")

elif page == "Evaluation Results":
    st.title("Evaluation Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{pipeline_metrics.get('precision', 0.0):.3f}")
    c2.metric("Recall", f"{pipeline_metrics.get('recall', 0.0):.3f}")
    c3.metric("F1", f"{pipeline_metrics.get('f1', 0.0):.3f}")
    c4.metric("AUC", f"{pipeline_metrics.get('auc', 0.0):.3f}")

    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    roc_path = os.path.join(outputs_dir, "roc_curve.png")
    risk_dist_path = os.path.join(outputs_dir, "risk_distribution.png")

    left, right = st.columns(2)
    if os.path.exists(cm_path):
        left.image(Image.open(cm_path), caption="Confusion Matrix")
    if os.path.exists(roc_path):
        right.image(Image.open(roc_path), caption="ROC Curve")
    if os.path.exists(risk_dist_path):
        st.image(Image.open(risk_dist_path), caption="Risk Score Distribution")

elif page == "Risk Scorecard":
    st.title("Risk Scorecard")
    scorecard_path = os.path.join(outputs_dir, "risk_scorecard.png")
    scatter_path = os.path.join(outputs_dir, "community_scatter.png")

    if os.path.exists(scorecard_path):
        st.image(Image.open(scorecard_path), caption="Top 20 communities by risk score")
    if os.path.exists(scatter_path):
        st.image(Image.open(scatter_path), caption="Illicit ratio vs community size")
