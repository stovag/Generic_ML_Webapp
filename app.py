import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    silhouette_score,
    roc_curve,
    roc_auc_score,
    normalized_mutual_info_score,
)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Set layout to wide in order to make the columns fill the whole page
st.set_page_config(layout="wide")


def run_supervised(algorithm, df, label_col, st_col):
    # Split dataframe to features and labels
    x = df.drop(label_col, axis=1)
    x = StandardScaler().fit_transform(x)
    y = df[label_col]

    # Split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Pick model
    if algorithm == "KNN":
        n_neighbors = st_col.number_input(
            label="Select minimum number of neighbors", min_value=1, value=5
        )
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif algorithm == "LR":
        model = LogisticRegression()
    elif algorithm == "DT":
        model = DecisionTreeClassifier(random_state=0)
    elif algorithm == "RF":
        max_depth = st_col.number_input(
            label="Select minimum number of neighbors", min_value=1, value=2
        )
        model = RandomForestClassifier(max_depth=max_depth, random_state=0)

    # Train and get test results
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calulcate metrics
    y_scores = model.predict_proba(x)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    results_dict = {
        "Metric": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy, precision, recall],
    }

    st_col.table(results_dict)

    y_onehot = pd.get_dummies(y, columns=model.classes_)

    # Draw AUC curve
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=700,
        height=500,
    )
    st_col.plotly_chart(fig, use_container_width=True)


def run_unsupervised(algorithm, df, label_col, st_col):
    x = df.drop(label_col, axis=1)

    # x = df
    x = StandardScaler().fit_transform(x)
    y = df[label_col]

    if algorithm == "DBS":
        eps = st_col.slider(
            label="Select maximum distance for neighbors",
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            value=0.5,
        )
        min_samples = st_col.number_input(
            label="Select minimum number of neighbors", min_value=3, value=5
        )

        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "MS":
        bandwidth = None
        if st_col.checkbox("Set Bandwidth"):
            bandwidth = st_col.slider(
                label="Select maximum distance for neighbors",
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                value=2.0,
            )

        model = MeanShift(bandwidth=bandwidth)
    elif algorithm == "SC":
        n_clusters = st_col.number_input(
            label="Select the number of clusters", min_value=1, value=2
        )
        model = SpectralClustering(
            n_clusters=n_clusters, assign_labels="discretize", random_state=0
        )
    elif algorithm == "AP":
        model = AffinityPropagation(random_state=5)

    # Fit the model to the data
    clustering_labels = model.fit_predict(x)
    graph_df = df.drop(label_col, axis=1)
    graph_df["labels"] = clustering_labels.astype("str")
    clusters_labeled = np.concatenate(
        [clustering_labels[:, np.newaxis], y[:, np.newaxis]], axis=1
    )

    silhouette = silhouette_score(graph_df, graph_df["labels"])
    nmis = normalized_mutual_info_score(y, clustering_labels)

    results_dict = {
        "Metric": ["Silhouette Score", "Normalized Mutual Information Score"],
        "Score": [silhouette, nmis],
    }

    st_col.table(results_dict)

    st_col.write(f"Number of Clusters Found: {len(set(clustering_labels))}")

    graph_df = graph_df.drop("labels", axis=1)
    n = len(graph_df.keys())

    # Plot the clusters
    with st_col.expander("Show Cluster Plot"):
        fig = px.scatter(
            x=graph_df[graph_df.keys()[0]],
            y=graph_df[graph_df.keys()[1]],
            color=clusters_labeled[:, 0],
            color_discrete_sequence=["orange", "red", "green", "blue", "purple"],
        )

        fig.update_layout(plot_bgcolor="rgb(47,47,47)")
        fig.update_layout(showlegend=False)
        fig.update_layout(height=400, width=550)
        st_col.plotly_chart(fig)


if __name__ == "__main__":
    file = st.sidebar.file_uploader("Insert CSV file...")
    if file is not None:
        sep = st.sidebar.text_input("Seperator: ", value=",")
        if st.sidebar.checkbox("File contains headers"):
            df = pd.read_csv(file, sep=sep)
        else:
            df = pd.read_csv(file, sep=sep, index_col=False)

        # Remove possible id column
        for key in df:
            if "id" in str.lower(key):
                df = df.drop(key, axis=1)

        if st.sidebar.checkbox("Show dataframe"):
            st.sidebar.write(df.head())

        label_col = st.sidebar.selectbox(
            "Select Labels Column", df.keys(), index=len(df.keys()) - 1
        )

        col1, col2 = st.columns((10, 10), gap="large")

        supervised_map = {
            "K Nearest Neighbors": "KNN",
            "Logistic Regression": "LR",
            "Decision Tree Classifier": "DT",
            "Random Forest Classifier": "RF",
        }

        supervised_classifier = col1.selectbox(
            "Pick a supervised algorithm to use", supervised_map.keys()
        )

        run_supervised(supervised_map[supervised_classifier], df, label_col, col1)

        unsupervised_map = {
            "Mean Shift": "MS",
            "DBSCAN": "DBS",
            "Spectral Clustering": "SC",
            "Affinity Propagation": "AP",
        }

        unsupervised_classifier = col2.selectbox(
            "Pick an unsupervised algorithm to use", unsupervised_map.keys()
        )

        run_unsupervised(unsupervised_map[unsupervised_classifier], df, label_col, col2)
