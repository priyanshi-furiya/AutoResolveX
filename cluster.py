import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import AzureOpenAI
import base64
from io import BytesIO
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- 1. Configuration ---
# <--- IMPORTANT: Replace with your Excel file path
EXCEL_FILE_PATH = "CaseDataWIthResolution.xlsx"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-02-01"  # Check your deployed API version
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_GPT4O_DEPLOYMENT")  # e.g., "gpt-35-turbo-16k"

# Validate Azure OpenAI credentials
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    print("Error: Azure OpenAI credentials not found in environment variables. Please check your .env file.")
    exit()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# --- 2. Data Loading and Preparation ---
print(f"Loading data from {EXCEL_FILE_PATH}...")
try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    print("Data loaded successfully.")
    print(f"Total tickets loaded: {len(df)}")
    print("\nAvailable columns in the Excel file:")
    print(df.columns.tolist())
except FileNotFoundError:
    print(f"Error: The file '{EXCEL_FILE_PATH}' was not found. Please ensure the path is correct.")
    exit()

# Handle potential missing values in text columns
available_text_columns = [col for col in ['Title', 'Description', 'Summary', 'Latest Comments', 'Task Type', 'Status', 'Ticket Owner', 'Assigned To'] if col in df.columns]
for col in available_text_columns:
    df[col] = df[col].fillna('')

# Initialize combined text column
df['combined_text'] = ''

# Add each available text column to the combined text
if 'Summary' in df.columns:
    df['combined_text'] += df['Summary'].astype(str) + ' '
if 'Latest Comments' in df.columns:
    df['combined_text'] += df['Latest Comments'].astype(str) + ' '
if 'Task Type' in df.columns:
    df['combined_text'] += 'Task Type: ' + df['Task Type'].astype(str) + ' '
if 'Status' in df.columns:
    df['combined_text'] += 'Status: ' + df['Status'].astype(str) + ' '
if 'Ticket Owner' in df.columns:
    df['combined_text'] += 'Owner: ' + df['Ticket Owner'].astype(str)

# Ensure there's no leading/trailing whitespace
df['combined_text'] = df['combined_text'].str.strip()

# Extract the combined descriptions for processing
incident_descriptions = df['combined_text'].tolist()

# Ensure there's data to process
if not incident_descriptions:
    print("No incident descriptions found to process. Exiting.")
    exit()

# Print dataset statistics
print("\n--- Dataset Statistics ---")
if 'Category' in df.columns:
    print("\nIncident Categories:")
    print(df['Category'].value_counts())

if 'Severity' in df.columns:
    print("\nSeverity Distribution:")
    print(df['Severity'].value_counts())

if 'Status' in df.columns:
    print("\nStatus Distribution:")
    print(df['Status'].value_counts())

print("-" * 30)

# --- 3. Text Preprocessing ---


def preprocess_text(text):
    text = str(text).lower()  # Ensure text is string and convert to lowercase
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


preprocessed_descriptions = [preprocess_text(
    desc) for desc in incident_descriptions]

print("\n--- Sample Preprocessed Descriptions ---")
for i, desc in enumerate(preprocessed_descriptions[:5]):  # Show first 5
    print(f"Original: {df['combined_text'].iloc[i]}")
    print(f"Preprocessed: {desc}\n")
print("-" * 30)

# --- 4. TF-IDF Vectorization ---
# You might adjust max_features, min_df, max_df based on your dataset size
vectorizer = TfidfVectorizer(
    max_features=2000, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(preprocessed_descriptions)

print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# --- 5. Clustering (K-Means) ---
# Determine an optimal number of clusters (K) - Experimentation is key!
# You could use the elbow method or silhouette score for more systematic K selection.
# For now, let's start with a reasonable guess based on common incident categories.
num_clusters = 5  # Adjust this value based on your data's expected natural groupings

kmeans = KMeans(n_clusters=num_clusters, random_state=42,
                n_init=10)  # n_init for robustness
kmeans.fit(tfidf_matrix)
# Add cluster labels back to the DataFrame
df['cluster_label'] = kmeans.labels_

print("\n--- Incidents with Cluster Labels (First 10) ---")
print(df[['Ticket Number', 'Summary', 'cluster_label']].head(10))
print("-" * 30)

# --- 6. Cluster Analysis with Azure OpenAI ---


def get_top_keywords(vectorizer, cluster_tfidf_scores, n=7):
    """Gets the top N keywords for a given cluster based on TF-IDF scores."""
    feature_names = vectorizer.get_feature_names_out()
    if cluster_tfidf_scores.ndim > 1:  # Handle case if scores is a matrix
        cluster_tfidf_scores = cluster_tfidf_scores.flatten()
    sorted_indices = cluster_tfidf_scores.argsort()[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:n]
                    if i < len(feature_names)]  # Ensure index is valid
    return top_keywords


def query_openai(prompt_text):
    """Helper function to query Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for IT incident root cause analysis. Be concise and actionable."},
                {"role": "user", "content": prompt_text}
            ],
            # Adjust for creativity (higher) vs. factual (lower)
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying Azure OpenAI: {e}")
        return "N/A"


def save_plot_to_base64():
    """Save matplotlib plot to base64 string"""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_base64


def generate_html_report(df, cluster_results, plot_base64=None):
    """Generate HTML report from clustering results"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Incident Cluster Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .cluster {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .keywords {{ color: #2c5282; }}
            .metrics {{ background-color: #f7fafc; padding: 10px; border-radius: 5px; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Incident Cluster Analysis Report</h1>
        <p>Generated on: {now}</p>
        <p>Total Incidents Analyzed: {len(df)}</p>

        <h2>Dataset Overview</h2>
        <div class="metrics">
            <h3>Category Distribution</h3>
            {df['Category'].value_counts().to_frame().to_html() if 'Category' in df.columns else 'No category data available'}

            <h3>Severity Distribution</h3>
            {df['Severity'].value_counts().to_frame().to_html() if 'Severity' in df.columns else 'No severity data available'}

            <h3>Status Distribution</h3>
            {df['Status'].value_counts().to_frame().to_html() if 'Status' in df.columns else 'No status data available'}
        </div>
    """

    # Add cluster information
    for cluster_id, results in cluster_results.items():
        html_content += f"""
        <div class="cluster">
            <h2>Cluster {cluster_id}</h2>
            <p><strong>Number of Incidents:</strong> {results['size']}</p>

            <h3>Keywords</h3>
            <p class="keywords">{', '.join(results['keywords'])}</p>

            <h3>Cluster Metadata</h3>
            <div class="metrics">
                {results['metadata']}
            </div>

            <h3>Temporal Analysis</h3>
            <div class="metrics">
                {results['temporal_analysis']}
            </div>

            <h3>Resolution Metrics</h3>
            <div class="metrics">
                {results['resolution_metrics']}
            </div>

            <h3>Cluster Analysis</h3>
            <p>{results['cluster_analysis']}</p>

            <h3>Root Cause Analysis</h3>
            <p>{results['root_cause_analysis']}</p>
        </div>
        """

    # Add visualization if available
    if plot_base64:
        html_content += f"""
        <div class="plot">
            <h2>Cluster Visualization</h2>
            <img src="data:image/png;base64,{plot_base64}" alt="Cluster Visualization">
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    return html_content

# Dictionary to store cluster results
cluster_results = {}

print("\n--- Analyzing Clusters with Azure OpenAI ---")
for i in range(num_clusters):
    cluster_df = df[df['cluster_label'] == i]
    cluster_descriptions = cluster_df['combined_text'].tolist()

    if cluster_df.empty:
        continue

    # Store cluster results
    cluster_results[i] = {
        'size': len(cluster_df),
        'keywords': [],
        'metadata': '',
        'temporal_analysis': '',
        'resolution_metrics': '',
        'cluster_analysis': '',
        'root_cause_analysis': ''
    }

    # Get and store keywords
    cluster_tfidf_sum = tfidf_matrix[df['cluster_label'] == i].sum(axis=0)
    top_keywords = get_top_keywords(vectorizer, cluster_tfidf_sum.A1, n=7)
    cluster_results[i]['keywords'] = top_keywords

    # Store metadata
    metadata_html = "<table>"
    metadata_cols = ['Category', 'Project', 'Severity', 'Priority', 'Status', 'Task Type']
    for col in metadata_cols:
        if col in cluster_df.columns:
            modes = cluster_df[col].mode().tolist()[:3]
            metadata_html += f"<tr><th>{col}</th><td>{', '.join(str(x) for x in modes)}</td></tr>"
    metadata_html += "</table>"
    cluster_results[i]['metadata'] = metadata_html

    # Store temporal analysis
    if 'Date Submitted' in cluster_df.columns:
        try:
            cluster_df['Date Submitted'] = pd.to_datetime(cluster_df['Date Submitted'])
            temporal_analysis = f"""
            <p>Earliest Incident: {cluster_df['Date Submitted'].min()}</p>
            <p>Latest Incident: {cluster_df['Date Submitted'].max()}</p>
            <p>Peak Month: {cluster_df['Date Submitted'].dt.to_period('M').mode().iloc[0]}</p>
            """
            cluster_results[i]['temporal_analysis'] = temporal_analysis
        except Exception:
            cluster_results[i]['temporal_analysis'] = "Could not analyze temporal data"

    # Get and store OpenAI analysis
    sample_descriptions = "\n- " + "\n- ".join(cluster_descriptions[:min(len(cluster_descriptions), 10)])

    summary_prompt = f"""Analyze these related incidents and identify:
    1. Common patterns and themes
    2. Potential systemic issues
    3. Key impacted systems or services

    Incident descriptions:
    {sample_descriptions}
    """
    cluster_results[i]['cluster_analysis'] = query_openai(summary_prompt)

    root_cause_prompt = f"""Based on these related incidents, identify:
    1. Root cause patterns
    2. Contributing factors
    3. Potential preventive measures

    Be specific and concise. Incident descriptions:
    {sample_descriptions}
    """
    cluster_results[i]['root_cause_analysis'] = query_openai(root_cause_prompt)

    # Store resolution metrics
    if 'Resolution Date' in cluster_df.columns and 'Date Submitted' in cluster_df.columns:
        try:
            cluster_df['Resolution Date'] = pd.to_datetime(cluster_df['Resolution Date'])
            cluster_df['Date Submitted'] = pd.to_datetime(cluster_df['Date Submitted'])
            cluster_df['Resolution Time'] = (cluster_df['Resolution Date'] - cluster_df['Date Submitted']).dt.total_seconds() / 3600
            resolution_metrics = f"""
            <p>Average Resolution Time: {cluster_df['Resolution Time'].mean():.2f} hours</p>
            <p>Median Resolution Time: {cluster_df['Resolution Time'].median():.2f} hours</p>
            """
            cluster_results[i]['resolution_metrics'] = resolution_metrics
        except Exception:
            cluster_results[i]['resolution_metrics'] = "Could not calculate resolution metrics"

# Generate visualization
plot_base64 = None
if tfidf_matrix.shape[0] > 1 and tfidf_matrix.shape[1] > 0:
    try:
        pca = PCA(n_components=2, random_state=42)
        reduced_data = pca.fit_transform(tfidf_matrix.toarray())

        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab10', num_clusters)

        for cluster_id in range(num_clusters):
            plt.scatter(
                reduced_data[df['cluster_label'] == cluster_id, 0],
                reduced_data[df['cluster_label'] == cluster_id, 1],
                color=colors(cluster_id),
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=100,
                edgecolors='w'
            )
        plt.title('Incident Clusters (PCA)', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_base64 = save_plot_to_base64()
    except Exception as e:
        print(f"Error during visualization: {e}")

# Generate and save HTML report
report_html = generate_html_report(df, cluster_results, plot_base64)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(os.path.dirname(__file__), f'cluster_analysis_report_{timestamp}.html')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_html)

print(f"\nAnalysis complete! Report saved to: {report_path}")
