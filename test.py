from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
from openai import AzureOpenAI
import numpy as np
import httpx
from functools import lru_cache
import time
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# --- Configuration ---
# WARNING: Hardcoding keys is NOT recommended for production. Use environment variables or Azure Key Vault.

# Cosmos DB Configuration
COSMOS_DB_URI = os.getenv('COSMOS_DB_URI')
COSMOS_DB_KEY = os.getenv('COSMOS_DB_KEY')
DATABASE_NAME = os.getenv('DATABASE_NAME')
CONTAINER_NAME = os.getenv('CONTAINER_NAME')

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
AZURE_OPENAI_GPT4O_DEPLOYMENT = os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

# Performance Configuration
COSMOS_FETCH_LIMIT_FOR_SIMILARITY = 300  # Increased back for better results
EMBEDDING_CACHE_SIZE = 1000  # Cache frequently used embeddings
SIMILARITY_THRESHOLD = 0.3   # Lowered threshold for more results
RECOMMENDATION_THRESHOLD = 0.4  # Separate threshold for recommendations
CONTEXT_THRESHOLD = 0.5      # Threshold for assistant context
MAX_WORKERS = 4              # For parallel processing

# SQLite Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'users.db')

# Flask and SQLAlchemy setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'


# Create tables
with app.app_context():
    db.create_all()

# --- Initialize Clients ---

# Azure Cosmos DB Client with optimized settings
try:
    cosmos_client_instance = CosmosClient(
        COSMOS_DB_URI,
        credential=COSMOS_DB_KEY,
        connection_timeout=100,  # Faster timeout
        request_timeout=300
    )
    cosmos_database = cosmos_client_instance.get_database_client(DATABASE_NAME)
    incidents_container = cosmos_database.get_container_client(CONTAINER_NAME)
    print("Successfully connected to Cosmos DB.")
except Exception as e:
    print(f"Error connecting to Cosmos DB: {e}")
    incidents_container = None

# Azure OpenAI Client with connection pooling
try:
    custom_http_client = httpx.Client(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )

    aoai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        http_client=custom_http_client
    )
    print("Successfully initialized Azure OpenAI client.")
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    aoai_client = None

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# In-memory cache for embeddings and incidents
embedding_cache = {}
# Reduced TTL to 3 minutes for more frequent updates
incidents_cache = {"data": [], "timestamp": 0, "ttl": 180}

# Simple user store - in production, use a proper database
users = {}

# Load and process ticket data


@lru_cache(maxsize=1)
def load_and_process_tickets():
    """Load and process ticket data from Excel file with caching."""
    df = pd.read_excel("CaseDataWIthResolution.xlsx")

    # Prepare text data
    df['combined_text'] = ''
    text_columns = {
        'Summary': '',
        'Latest Comments': '',
        'Task Type': 'Task Type: ',
        'Status': 'Status: '
    }

    for col, prefix in text_columns.items():
        if col in df.columns:
            df['combined_text'] += prefix + df[col].astype(str) + ' '

    df['combined_text'] = df['combined_text'].str.strip()
    return df


def perform_cluster_analysis(df, num_clusters=5):
    """Perform cluster analysis on ticket data."""
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    df['cluster_label'] = kmeans.labels_

    return df, tfidf_matrix, vectorizer


def analyze_cluster(cluster_df, cluster_id, tfidf_matrix, vectorizer, df_labels):
    """Analyze a single cluster and return its metrics and analysis."""
    # Get keywords
    cluster_tfidf_sum = tfidf_matrix[df_labels == cluster_id].sum(axis=0)
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = cluster_tfidf_sum.A1.argsort()[::-1]
    keywords = [feature_names[idx] for idx in sorted_indices[:7]]

    # Calculate metrics
    metrics = {
        'avg_time': 'N/A',
        'median_time': 'N/A',
        'temporal': 'Temporal data not available'
    }

    try:
        if all(col in cluster_df.columns for col in ['Resolution Date', 'Date Submitted']):
            cluster_df['Resolution Date'] = pd.to_datetime(
                cluster_df['Resolution Date'])
            cluster_df['Date Submitted'] = pd.to_datetime(
                cluster_df['Date Submitted'])
            resolution_times = (
                cluster_df['Resolution Date'] - cluster_df['Date Submitted']).dt.total_seconds() / 3600
            metrics['avg_time'] = f"{resolution_times.mean():.1f}"
            metrics['median_time'] = f"{resolution_times.median():.1f}"

            # Temporal analysis
            metrics['temporal'] = (
                f"Earliest: {cluster_df['Date Submitted'].min().strftime('%Y-%m-%d')}\n"
                f"Latest: {cluster_df['Date Submitted'].max().strftime('%Y-%m-%d')}\n"
                f"Peak Month: {cluster_df['Date Submitted'].dt.to_period('M').mode().iloc[0]}"
            )
    except Exception as e:
        print(f"Error calculating metrics: {e}")

    # Get cluster metadata
    metadata = {
        'size': len(cluster_df),
        'categories': cluster_df['Category'].value_counts().to_dict() if 'Category' in cluster_df.columns else {},
        'task_types': cluster_df['Task Type'].value_counts().to_dict() if 'Task Type' in cluster_df.columns else {},
        'severities': cluster_df['Severity'].value_counts().to_dict() if 'Severity' in cluster_df.columns else {},
        'keywords': keywords,
        'temporal': metrics['temporal']
    }

    # Get cluster analysis from OpenAI
    analysis = get_cluster_analysis(cluster_df, metadata, metrics)

    return {
        'id': cluster_id,
        'size': metadata['size'],
        'keywords': keywords,
        'avg_resolution_time': metrics['avg_time'],
        'median_resolution_time': metrics['median_time'],
        'temporal_analysis': metrics['temporal'],
        'root_cause_analysis': analysis
    }


def get_cluster_analysis(cluster_df, metadata, metrics):
    """Get cluster analysis from OpenAI."""
    try:
        sample_tickets = cluster_df['combined_text'].head(5).tolist()
        prompt = f"""As an IT incident analysis expert, analyze this cluster of related incidents:

        Cluster Statistics:
        - Total Incidents: {metadata['size']}
        - Average Resolution: {metrics['avg_time']} hours
        - Median Resolution: {metrics['median_time']} hours
        - Categories: {', '.join(f'{k}({v})' for k,v in metadata['categories'].items())}
        - Task Types: {', '.join(f'{k}({v})' for k,v in metadata['task_types'].items())}
        - Severity Levels: {', '.join(f'{k}({v})' for k,v in metadata['severities'].items())}
        - Key Terms: {', '.join(metadata['keywords'])}
        - Time Range: {metadata['temporal']}

        Sample Incidents:
        {chr(10).join(f'- {ticket}' for ticket in sample_tickets)}

        Provide a comprehensive root cause analysis including:
        1. Primary Root Causes: Identify the fundamental issues causing these incidents
        2. Systemic Patterns: Note any recurring patterns or systemic issues
        3. Impact Areas: List major systems, services, or business areas affected
        4. Resolution Patterns: Common successful resolution approaches
        5. Preventive Measures: Specific actions to prevent recurrence
        6. Process Improvements: Recommendations for process/system enhancements

        Focus on actionable insights and clear patterns."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": """You are an expert IT incident analyst specializing in root cause analysis. 
                Your analysis should be detailed, data-driven, and focused on actionable insights. 
                Structure your response clearly with headings and bullet points.
                Prioritize identifying systemic issues and preventive measures."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )

        analysis = response.choices[0].message.content
        if not analysis or analysis.strip() == "":
            return "Analysis not available"

        # Format the analysis for better readability
        for i in range(1, 7):
            analysis = analysis.replace(f"{i}.", f"\n{i}.")

        return analysis
    except Exception as e:
        print(f"Error in OpenAI analysis: {e}")
        return "Analysis not available - Service error. Please try again later."


# --- Helper Functions ---

@lru_cache(maxsize=1)
def get_tickets():
    """Get tickets data from Excel file with caching."""
    try:
        df = pd.read_excel("CaseDataWIthResolution.xlsx")
        tickets = df.to_dict('records')
        return tickets
    except Exception as e:
        print(f"Error loading tickets: {e}")
        return []


@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def get_embedding_cached(text: str, model: str = AZURE_OPENAI_EMBEDDING_DEPLOYMENT):
    """Cached version of embedding generation."""
    return get_embedding_internal(text, model)


def get_embedding_internal(text: str, model: str):
    """Internal embedding function without caching."""
    if not aoai_client:
        print("Error: Azure OpenAI client not initialized for embeddings.")
        return None
    if not text or not text.strip():
        print("Error: Input text for embedding is empty.")
        return None
    try:
        # Truncate very long text to avoid API limits
        text = text[:8000]  # OpenAI embedding models have token limits
        response = aoai_client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text[:50]}...': {e}")
        return None


def get_embedding(text: str, model: str = AZURE_OPENAI_EMBEDDING_DEPLOYMENT):
    """Get embedding with caching support."""
    # Create a cache key
    cache_key = f"{text[:500]}_{model}"  # Limit key size

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    embedding = get_embedding_internal(text, model)

    # Cache the result (with size limit)
    if len(embedding_cache) < EMBEDDING_CACHE_SIZE:
        embedding_cache[cache_key] = embedding

    return embedding


def cosine_similarity_fast(vec1, vec2):
    """Optimized cosine similarity calculation."""
    if vec1 is None or vec2 is None:
        return 0.0

    # Convert to numpy arrays once
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    # Fast computation using numpy
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norms == 0:
        return 0.0

    return float(dot_product / norms)


def fetch_incidents_cached():
    """Fetch incidents with caching."""
    current_time = time.time()

    # Check if cache is valid
    if (incidents_cache["data"] and
            current_time - incidents_cache["timestamp"] < incidents_cache["ttl"]):
        print(f"Using cached incidents ({len(incidents_cache['data'])} items)")
        return incidents_cache["data"]

    # Fetch fresh data
    print("Fetching fresh incidents from Cosmos DB...")
    incidents = fetch_incidents_for_similarity_optimized()

    # Update cache
    incidents_cache["data"] = incidents
    incidents_cache["timestamp"] = current_time

    return incidents


def fetch_incidents_for_similarity_optimized(limit=COSMOS_FETCH_LIMIT_FOR_SIMILARITY):
    """Optimized incident fetching with better query."""
    if not incidents_container:
        print("Error: Cosmos DB container not initialized.")
        return []

    try:
        # More efficient query - only select needed fields
        query = f"""
        SELECT c.id, c["Ticket Number"], c.Summary, c.Project, c.Category, 
               c.Severity, c.Priority, c.Status, c["Resolution Date"], 
               c["Latest Comments"], c._ts
        FROM c 
        WHERE IS_DEFINED(c.Summary) AND LENGTH(c.Summary) > 5
        ORDER BY c._ts DESC 
        OFFSET 0 LIMIT {limit}
        """

        items = list(incidents_container.query_items(
            query=query,
            enable_cross_partition_query=True,
            max_item_count=limit
        ))

        print(f"Fetched {len(items)} incidents from Cosmos DB")
        return items

    except Exception as e:
        print(f"Error fetching incidents from Cosmos DB: {e}")
        return []


def calculate_similarity_parallel(query_vector, incidents):
    """Calculate similarities in parallel."""
    def calculate_single_similarity(incident):
        try:
            incident_summary = incident.get("Summary", "")
            if not incident_summary.strip():
                return None

            incident_vector = get_embedding(incident_summary)
            if incident_vector:
                similarity = cosine_similarity_fast(
                    query_vector, incident_vector)

                # Only return if above threshold
                if similarity >= SIMILARITY_THRESHOLD:
                    return {
                        "TicketNumber": incident.get("Ticket Number"),
                        "Summary": incident_summary,
                        "Project": incident.get("Project"),
                        "Category": incident.get("Category"),
                        "Severity": incident.get("Severity"),
                        "Priority": incident.get("Priority"),
                        "Status": incident.get("Status"),
                        "ResolutionDate": incident.get("Resolution Date"),
                        "LatestComments": incident.get("Latest Comments"),
                        "id": incident.get("id"),
                        "similarity_score": float(similarity)
                    }
        except Exception as e:
            print(
                f"Error processing incident {incident.get('id', 'unknown')}: {e}")
        return None

    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(calculate_single_similarity, incidents))

    # Filter out None results and sort
    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x['similarity_score'], reverse=True)

    return valid_results

# --- API Endpoints ---


@app.route('/search-similar-incidents', methods=['POST'])
def search_similar_incidents_route():
    """Optimized similar incidents search."""
    start_time = time.time()

    if not incidents_container or not aoai_client:
        return jsonify({"error": "Backend services not configured properly."}), 503

    try:
        data = request.get_json()
        if not data or 'text' not in data or not data['text'].strip():
            return jsonify({"error": "Missing or empty 'text' in request body"}), 400

        query_text = data['text']

        # Get embedding for query
        embedding_start = time.time()
        query_vector = get_embedding(query_text)
        embedding_time = time.time() - embedding_start

        if not query_vector:
            return jsonify({"error": "Failed to generate embedding for query text."}), 500

        # Fetch incidents (cached)
        fetch_start = time.time()
        historical_incidents = fetch_incidents_cached()
        fetch_time = time.time() - fetch_start

        if not historical_incidents:
            return jsonify({
                "message": "No historical incidents found.",
                "similar_incidents": []
            }), 200

        # Calculate similarities in parallel
        similarity_start = time.time()
        similar_incidents = calculate_similarity_parallel(
            query_vector, historical_incidents)
        similarity_time = time.time() - similarity_start

        total_time = time.time() - start_time

        print(f"Performance metrics - Total: {total_time:.2f}s, "
              f"Embedding: {embedding_time:.2f}s, "
              f"Fetch: {fetch_time:.2f}s, "
              f"Similarity: {similarity_time:.2f}s")

        # Return top 5 results
        return jsonify(similar_incidents[:5]), 200

    except Exception as e:
        print(f"Error in /search-similar-incidents: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/recommend-resolution', methods=['POST'])
def recommend_resolution_route():
    """Optimized resolution recommendation."""
    start_time = time.time()

    if not incidents_container or not aoai_client:
        return jsonify({"error": "Backend services not configured properly."}), 503

    try:
        data = request.get_json()
        if not data or 'summary' not in data or not data['summary'].strip():
            return jsonify({"error": "Missing or empty 'summary' in request body"}), 400

        incident_text = data['summary']
        if 'description' in data and data['description'].strip():
            incident_text += " " + data['description']

        query_vector = get_embedding(incident_text)
        if not query_vector:
            return jsonify({"error": "Failed to generate embedding for incident text."}), 500

        historical_incidents = fetch_incidents_cached()
        if not historical_incidents:
            return jsonify({
                "message": "No historical incidents found for recommendation.",
                "recommendations": []
            }), 200

        # Calculate similarities with resolution focus
        recommendations = []

        def process_incident_for_recommendation(incident):
            try:
                incident_summary = incident.get("Summary", "")
                if not incident_summary.strip():
                    return None

                incident_vector = get_embedding(incident_summary)
                if incident_vector:
                    similarity = cosine_similarity_fast(
                        query_vector, incident_vector)

                    if similarity >= RECOMMENDATION_THRESHOLD:
                        resolution_text = incident.get(
                            "LatestComments", "No resolution information found.")
                        if not resolution_text or not resolution_text.strip():
                            resolution_text = "No explicit resolution information found."

                        return {
                            "ticket_info": {
                                "TicketNumber": incident.get("Ticket Number"),
                                "Summary": incident_summary,
                                "Project": incident.get("Project"),
                                "Category": incident.get("Category"),
                                "id": incident.get("id")
                            },
                            "probable_resolution": resolution_text,
                            "similarity_score": float(similarity)
                        }
            except Exception as e:
                print(f"Error processing incident for recommendation: {e}")
            return None

        # Process in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(
                process_incident_for_recommendation, historical_incidents))

        recommendations = [r for r in results if r is not None]
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)

        total_time = time.time() - start_time
        print(f"Recommendation processing time: {total_time:.2f}s")

        return jsonify({"recommendations": recommendations[:3]}), 200

    except Exception as e:
        print(f"Error in /recommend-resolution: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/ask-assistant', methods=['POST'])
def ask_assistant_route():
    """Optimized AI assistant with enhanced context retrieval."""
    if not aoai_client:
        return jsonify({"error": "Azure OpenAI client not configured for chat."}), 503

    try:
        data = request.get_json()
        if not data or 'query' not in data or not data['query'].strip():
            return jsonify({"error": "Missing or empty 'query' in request body"}), 400

        user_query = data['query']
        current_incident_context_data = data.get(
            'current_incident_context', None)

        context_for_llm_str = ""
        context_found = False

        # Always try to find context from historical incidents
        if incidents_container:
            search_text = user_query

            # Prioritize current incident context if provided
            if current_incident_context_data and 'summary' in current_incident_context_data:
                search_text = current_incident_context_data['summary']
                if 'description' in current_incident_context_data:
                    search_text += " " + \
                        current_incident_context_data.get('description', '')

            print(f"Searching for context with text: {search_text[:100]}...")

            if search_text.strip():
                context_start_time = time.time()
                context_vector = get_embedding(search_text)

                if context_vector:
                    # Use cached incidents for context retrieval
                    cached_incidents = fetch_incidents_cached()
                    print(
                        f"Processing {len(cached_incidents)} cached incidents for context...")

                    # Process more incidents for better context
                    context_incidents = []
                    processed_count = 0

                    for incident in cached_incidents:
                        summary = incident.get("Summary", "")
                        if summary.strip():
                            try:
                                inc_vector = get_embedding(summary)
                                if inc_vector:
                                    sim = cosine_similarity_fast(
                                        context_vector, inc_vector)
                                    processed_count += 1

                                    # Lower threshold for context to get more results
                                    if sim >= CONTEXT_THRESHOLD:
                                        context_incidents.append({
                                            "ticket": incident.get("Ticket Number", "N/A"),
                                            # Increased length
                                            "summary": summary[:200],
                                            # Increased length
                                            "resolution": incident.get("Latest Comments", "N/A")[:300],
                                            "category": incident.get("Category", "N/A"),
                                            "similarity": sim
                                        })
                                        context_found = True
                            except Exception as e:
                                print(
                                    f"Error processing incident for context: {e}")
                                continue

                    print(
                        f"Processed {processed_count} incidents, found {len(context_incidents)} with similarity >= {CONTEXT_THRESHOLD}")

                    # If no high-similarity matches, lower the threshold temporarily
                    if len(context_incidents) == 0 and processed_count > 0:
                        print(
                            "No high-similarity matches found, trying lower threshold...")
                        # Check top 100 with lower threshold
                        for incident in cached_incidents[:100]:
                            summary = incident.get("Summary", "")
                            if summary.strip():
                                try:
                                    inc_vector = get_embedding(summary)
                                    if inc_vector:
                                        sim = cosine_similarity_fast(
                                            context_vector, inc_vector)
                                        if sim >= 0.3:  # Much lower threshold
                                            context_incidents.append({
                                                "ticket": incident.get("Ticket Number", "N/A"),
                                                "summary": summary[:200],
                                                "resolution": incident.get("Latest Comments", "N/A")[:300],
                                                "category": incident.get("Category", "N/A"),
                                                "similarity": sim
                                            })
                                            context_found = True
                                            if len(context_incidents) >= 3:  # Stop after finding 3
                                                break
                                except Exception as e:
                                    continue

                    context_incidents.sort(
                        key=lambda x: x['similarity'], reverse=True)

                    if context_incidents:
                        context_items = []
                        # Top 3 for context
                        for i, item in enumerate(context_incidents[:3]):
                            context_items.append(
                                f"{i+1}. Ticket {item['ticket']} (Category: {item['category']}, Similarity: {item['similarity']:.2f}):\n"
                                f"   Summary: {item['summary']}\n"
                                f"   Resolution/Comments: {item['resolution']}\n"
                            )
                        context_for_llm_str = "Relevant past incidents from knowledge base:\n" + \
                            "\n".join(context_items)
                        print(
                            f"Generated context from {len(context_incidents)} incidents")

                    context_time = time.time() - context_start_time
                    print(f"Context retrieval took {context_time:.2f}s")
                else:
                    print("Failed to generate embedding for context search")

        # Always provide a comprehensive system prompt
        system_prompt = (
            "You are an expert IT support assistant with access to a knowledge base of past incidents. "
            "Your role is to help support agents resolve issues quickly and effectively. "
            "When provided with context from past incidents, use that information to give specific, "
            "actionable advice. If no specific context is available, provide general best practices "
            "and troubleshooting steps. Always be practical and solution-focused."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add context if found
        if context_for_llm_str:
            messages.append(
                {"role": "system", "content": f"Knowledge Base Context:\n{context_for_llm_str}"})
        else:
            messages.append(
                {"role": "system", "content": "No specific historical incidents found in knowledge base for this query. Provide general troubleshooting advice."})

        # Add current incident context if provided
        if current_incident_context_data:
            incident_context = f"Current Incident Details:\n"
            for key, value in current_incident_context_data.items():
                incident_context += f"- {key}: {value}\n"
            messages.append({"role": "system", "content": incident_context})

        messages.append({"role": "user", "content": user_query})

        # Generate response
        chat_response = aoai_client.chat.completions.create(
            model=AZURE_OPENAI_GPT4O_DEPLOYMENT,
            messages=messages,
            max_tokens=800,  # Increased for more detailed responses
            temperature=0.3
        )

        assistant_answer = chat_response.choices[0].message.content

        return jsonify({
            "answer": assistant_answer,
            "context_used": context_found,
            "context_summary": context_for_llm_str if context_for_llm_str else "No relevant historical incidents found",
            "incidents_processed": len(cached_incidents) if incidents_container else 0
        }), 200

    except Exception as e:
        print(f"Error in /ask-assistant: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/ask-assistant')
def ask_assistant_page():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('ask-assistant.html')

# --- Performance monitoring endpoint ---


@app.route('/health', methods=['GET'])
def health_check():
    """Health check with performance info."""
    return jsonify({
        "status": "healthy",
        "cosmos_connected": incidents_container is not None,
        "openai_connected": aoai_client is not None,
        "cached_incidents": len(incidents_cache["data"]),
        "embedding_cache_size": len(embedding_cache)
    }), 200

# --- Cache management endpoints ---


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all caches."""
    global embedding_cache, incidents_cache
    embedding_cache.clear()
    incidents_cache = {"data": [], "timestamp": 0, "ttl": 180}
    return jsonify({"message": "Caches cleared successfully"}), 200


@app.route('/debug-data', methods=['GET'])
def debug_data():
    """Debug endpoint to check what data is available."""
    try:
        cached_incidents = fetch_incidents_cached()

        # Sample data analysis
        total_incidents = len(cached_incidents)
        incidents_with_summary = sum(
            1 for inc in cached_incidents if inc.get("Summary", "").strip())
        incidents_with_comments = sum(
            1 for inc in cached_incidents if inc.get("Latest Comments", "").strip())

        # Sample incident data (first 3)
        sample_incidents = []
        for i, incident in enumerate(cached_incidents[:3]):
            sample_incidents.append({
                "index": i,
                "ticket_number": incident.get("Ticket Number", "N/A"),
                "summary": incident.get("Summary", "")[:100] + "..." if len(incident.get("Summary", "")) > 100 else incident.get("Summary", ""),
                "has_comments": bool(incident.get("Latest Comments", "").strip()),
                "category": incident.get("Category", "N/A")
            })

        return jsonify({
            "total_incidents": total_incidents,
            "incidents_with_summary": incidents_with_summary,
            "incidents_with_comments": incidents_with_comments,
            "cache_age_seconds": time.time() - incidents_cache["timestamp"] if incidents_cache["timestamp"] else 0,
            "sample_incidents": sample_incidents,
            "thresholds": {
                "similarity": SIMILARITY_THRESHOLD,
                "recommendation": RECOMMENDATION_THRESHOLD,
                "context": CONTEXT_THRESHOLD
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('similar_tickets'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_email'] = user.email
            session['user_name'] = user.name
            return redirect(url_for('similar_tickets'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists')
        else:
            try:
                hashed_password = generate_password_hash(password)
                new_user = User(name=name, email=email,
                                password=hashed_password)
                db.session.add(new_user)
                db.session.commit()

                session['user_email'] = email
                session['user_name'] = name
                return redirect(url_for('similar_tickets'))
            except Exception as e:
                db.session.rollback()
                flash('An error occurred during registration. Please try again.')
                print(f"Registration error: {e}")

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('login'))


@app.route('/similar-tickets')
def similar_tickets():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    # Get all tickets initially, showing most recent first
    tickets = get_tickets()
    similar = tickets[:5] if tickets else []
    return render_template('similar-tickets.html', tickets=similar)


@app.route('/recommended-tickets')
def recommended_tickets():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    # Get tickets with high priority or severity
    tickets = get_tickets()
    recommended = [t for t in tickets if str(t.get('Priority', '')).lower() == 'high'
                   or str(t.get('Severity', '')).lower() in ['severity 1', 'severity 2']][:5]
    return render_template('recommended-tickets.html', tickets=recommended)


@app.route('/ticket-details/<ticket_id>')
def ticket_details(ticket_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))

    tickets = get_tickets()
    ticket = next((t for t in tickets if str(
        t.get('Ticket Number')) == ticket_id), None)
    if ticket is None:
        flash('Ticket not found')
        return redirect(url_for('similar_tickets'))

    return render_template('ticket-details.html', ticket=ticket)


@app.route('/search')
def search():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    query = request.args.get('q', '').lower()
    if query:
        tickets = get_tickets()
        search_results = [
            t for t in tickets if query in str(t.get('Summary', '')).lower()]
        return jsonify(search_results)
    return jsonify([])


# --- Cluster Analysis Functions ---

def load_and_process_tickets():
    """Load and process ticket data from Excel file."""
    try:
        df = pd.read_excel("CaseDataWIthResolution.xlsx")

        # Prepare text data
        df['combined_text'] = ''
        text_columns = {
            'Summary': '',
            'Latest Comments': '',
            'Task Type': 'Task Type: ',
            'Status': 'Status: '
        }

        for col, prefix in text_columns.items():
            if col in df.columns:
                df['combined_text'] += prefix + df[col].astype(str) + ' '

        df['combined_text'] = df['combined_text'].str.strip()
        return df
    except Exception as e:
        print(f"Error loading ticket data: {e}")
        return None


def perform_cluster_analysis(df, num_clusters=5):
    """Perform cluster analysis on ticket data."""
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85
        )
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

        # Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)
        df['cluster_label'] = kmeans.labels_

        return df, tfidf_matrix, vectorizer
    except Exception as e:
        print(f"Error in cluster analysis: {e}")
        return None, None, None


def get_visualization(df, tfidf_matrix, num_clusters):
    """Generate cluster visualization."""
    try:
        pca = PCA(n_components=2, random_state=42)
        reduced_data = pca.fit_transform(tfidf_matrix.toarray())

        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab10', num_clusters)

        for cluster_id in range(num_clusters):
            mask = df['cluster_label'] == cluster_id
            plt.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                color=colors(cluster_id),
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=100,
                edgecolors='w'
            )

        plt.title('Incident Clusters Visualization', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(title='Cluster ID', bbox_to_anchor=(
            1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return plot_base64
    except Exception as e:
        print(f"Error in visualization: {e}")
        return None


def analyze_cluster(cluster_df, cluster_id, tfidf_matrix, vectorizer, labels):
    """Analyze a single cluster and return its metrics."""
    try:
        # Get keywords
        cluster_tfidf_sum = tfidf_matrix[labels == cluster_id].sum(axis=0)
        feature_names = vectorizer.get_feature_names_out()
        sorted_indices = cluster_tfidf_sum.A1.argsort()[::-1]
        keywords = [feature_names[idx] for idx in sorted_indices[:7]]

        # Calculate metrics
        metrics = calculate_cluster_metrics(cluster_df)

        # Get cluster metadata
        metadata = get_cluster_metadata(cluster_df, keywords)

        # Get OpenAI analysis
        root_cause = get_root_cause_analysis(cluster_df, metadata, metrics)

        return {
            'id': cluster_id,
            'size': len(cluster_df),
            'keywords': keywords,
            'avg_resolution_time': metrics['avg_time'],
            'median_resolution_time': metrics['median_time'],
            'temporal_analysis': metrics['temporal'],
            'root_cause_analysis': root_cause
        }
    except Exception as e:
        print(f"Error analyzing cluster {cluster_id}: {e}")
        return None


def calculate_cluster_metrics(cluster_df):
    """Calculate cluster metrics."""
    metrics = {
        'avg_time': 'N/A',
        'median_time': 'N/A',
        'temporal': 'Temporal data not available'
    }

    try:
        if all(col in cluster_df.columns for col in ['Resolution Date', 'Date Submitted']):
            cluster_df['Resolution Date'] = pd.to_datetime(
                cluster_df['Resolution Date'])
            cluster_df['Date Submitted'] = pd.to_datetime(
                cluster_df['Date Submitted'])
            resolution_times = (
                cluster_df['Resolution Date'] - cluster_df['Date Submitted']).dt.total_seconds() / 3600

            metrics.update({
                'avg_time': f"{resolution_times.mean():.1f}",
                'median_time': f"{resolution_times.median():.1f}",
                'temporal': (
                    f"Earliest: {cluster_df['Date Submitted'].min().strftime('%Y-%m-%d')}\n"
                    f"Latest: {cluster_df['Date Submitted'].max().strftime('%Y-%m-%d')}\n"
                    f"Peak Month: {cluster_df['Date Submitted'].dt.to_period('M').mode().iloc[0]}"
                )
            })
    except Exception as e:
        print(f"Error calculating metrics: {e}")

    return metrics


def get_cluster_metadata(cluster_df, keywords):
    """Get cluster metadata."""
    return {
        'size': len(cluster_df),
        'categories': cluster_df['Category'].value_counts().to_dict() if 'Category' in cluster_df.columns else {},
        'task_types': cluster_df['Task Type'].value_counts().to_dict() if 'Task Type' in cluster_df.columns else {},
        'severities': cluster_df['Severity'].value_counts().to_dict() if 'Severity' in cluster_df.columns else {},
        'keywords': keywords
    }


def get_root_cause_analysis(cluster_df, metadata, metrics):
    """Get root cause analysis from OpenAI."""
    try:
        sample_tickets = cluster_df['combined_text'].head(5).tolist()
        prompt = f"""As an IT incident analysis expert, analyze this cluster of incidents:

        Stats:
        - Total Incidents: {metadata['size']}
        - Avg Resolution: {metrics['avg_time']} hours
        - Median Resolution: {metrics['median_time']} hours
        - Categories: {', '.join(f'{k}({v})' for k,v in metadata['categories'].items())}
        - Task Types: {', '.join(f'{k}({v})' for k,v in metadata['task_types'].items())}
        - Severities: {', '.join(f'{k}({v})' for k,v in metadata['severities'].items())}
        - Key Terms: {', '.join(metadata['keywords'])}

        Sample Incidents:
        {chr(10).join(f'- {ticket}' for ticket in sample_tickets)}

        Analyze:
        1. Root Causes
        2. Patterns
        3. Impact Areas
        4. Resolution Steps
        5. Prevention
        6. Improvements
        """

        response = aoai_client.chat.completions.create(
            model=AZURE_OPENAI_GPT4O_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "You are an IT incident analyst. Provide detailed, actionable insights."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        analysis = response.choices[0].message.content
        if not analysis or analysis.strip() == "":
            return "Analysis not available"

        # Format analysis
        for i in range(1, 7):
            analysis = analysis.replace(f"{i}.", f"\n{i}.")

        return analysis
    except Exception as e:
        print(f"Error in OpenAI analysis: {e}")
        return "Analysis not available"


@app.route('/root-cause-pattern')
def root_cause_pattern():
    """Route handler for root cause pattern analysis."""
    if 'user_email' not in session:
        return redirect(url_for('login'))

    try:
        # Load data
        df = load_and_process_tickets()
        if df is None:
            flash('Error loading ticket data')
            return render_template('root-cause-pattern.html', clusters=[], plot_base64=None)

        # Perform clustering
        df, tfidf_matrix, vectorizer = perform_cluster_analysis(df)
        if any(x is None for x in [df, tfidf_matrix, vectorizer]):
            flash('Error performing cluster analysis')
            return render_template('root-cause-pattern.html', clusters=[], plot_base64=None)

        # Analyze clusters
        num_clusters = 5  # Fixed number of clusters
        clusters = []

        for i in range(num_clusters):
            cluster_df = df[df['cluster_label'] == i]
            if len(cluster_df) > 0:
                cluster_result = analyze_cluster(
                    cluster_df, i, tfidf_matrix, vectorizer, df['cluster_label'].values
                )
                if cluster_result:
                    clusters.append(cluster_result)

        if not clusters:
            flash('No meaningful clusters found')
            return render_template('root-cause-pattern.html', clusters=[], plot_base64=None)

        # Generate visualization
        plot_base64 = get_visualization(df, tfidf_matrix, num_clusters)

        return render_template('root-cause-pattern.html', clusters=clusters, plot_base64=plot_base64)

    except Exception as e:
        print(f"Error in root cause pattern analysis: {e}")
        flash('Error analyzing root cause patterns')
        return render_template('root-cause-pattern.html', clusters=[], plot_base64=None)


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting optimized Flask server...")
    print(f"Performance settings:")
    print(f"- Cosmos fetch limit: {COSMOS_FETCH_LIMIT_FOR_SIMILARITY}")
    print(f"- Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"- Recommendation threshold: {RECOMMENDATION_THRESHOLD}")
    print(f"- Context threshold: {CONTEXT_THRESHOLD}")
    print(f"- Embedding cache size: {EMBEDDING_CACHE_SIZE}")
    print(f"- Max workers: {MAX_WORKERS}")
    print(f"- Cache TTL: {incidents_cache['ttl']} seconds")

    if not incidents_container:
        print("WARNING: Cosmos DB client not initialized.")
    if not aoai_client:
        print("WARNING: Azure OpenAI client not initialized.")

    app.run(debug=True, port=5000, threaded=True)
