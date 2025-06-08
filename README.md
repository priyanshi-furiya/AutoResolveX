# AutoResolveX

**AI-Powered IT Ticket Triage and Resolution Engine**

AutoResolveX is an intelligent IT service management platform that leverages Azure OpenAI to automatically categorize, analyze, and recommend solutions for IT tickets. The application features a Microsoft Teams bot integration and provides real-time insights through advanced clustering and similarity analysis.

## 🌐 Live Application

**Application URL:** https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net

- **Login Page:** https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net/login

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [API Endpoints](#api-endpoints)
- [Teams Bot Integration](#teams-bot-integration)
- [Usage Guide](#usage-guide)
- [Deployment](#deployment)

## ✨ Features

### Core Functionality
- **🤖 AI-Powered Ticket Analysis**: Leverages Azure OpenAI GPT-4 for intelligent ticket categorization and solution recommendations
- **🔍 Similar Ticket Search**: Advanced vector similarity search using Azure OpenAI embeddings
- **📊 Cluster Analysis**: Automated ticket clustering with root cause pattern identification
- **💬 AI Assistant**: Contextual chat interface for ticket-specific assistance
- **🔗 Direct Access Links**: Secure token-based access to tickets without authentication

### Microsoft Teams Integration
- **Teams Bot**: Native Microsoft Teams bot with natural language processing
- **Slash Commands**: Quick access to ticket operations via `/similar`, `/recommend`, `/help`
- **Direct Links**: Generate secure links for sharing tickets in Teams conversations

### Analytics & Insights
- **Real-time Dashboards**: Performance metrics and ticket analytics
- **Root Cause Analysis**: AI-generated insights into recurring ticket patterns
- **Temporal Analysis**: Time-based incident tracking and trending

## 🏗️ Architecture

### Technology Stack
- **Backend**: Flask (Python 3.11)
- **Database**: Azure Cosmos DB (NoSQL)
- **AI Services**: Azure OpenAI (GPT-4, Text-Embedding-Ada-002)
- **Bot Framework**: Microsoft Bot Framework SDK
- **Frontend**: HTML5, CSS3, JavaScript with Marked.js for markdown rendering
- **Deployment**: Azure App Service (Linux containers)

### Key Components
- **Flask Application** (`test.py`): Main web application server
- **Teams Bot** (`bot.py`): Microsoft Teams integration handler
- **Cluster Analysis** (`cluster.py`): Machine learning clustering engine
- **Utilities** (`utils.py`): Helper functions and security utilities

## 📋 Prerequisites

- Python 3.11+
- Azure subscription with:
  - Azure OpenAI service
  - Azure Cosmos DB account
  - Azure App Service (for deployment)
  - Azure Bot Service (for Teams integration)
- Microsoft Teams Developer Account (for bot deployment)
- Git (for version control)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AutoResolveX
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
```

## ⚙️ Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Azure Cosmos DB Configuration
COSMOS_DB_URI=https://your-cosmosdb-account.documents.azure.com:443/
COSMOS_DB_KEY=your-cosmos-db-primary-key
DATABASE_NAME=your-database-name
CONTAINER_NAME=your-container-name

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01

# Microsoft Bot Framework (for Teams integration)
MICROSOFT_APP_ID=your-microsoft-app-id
MICROSOFT_APP_PASSWORD=your-microsoft-app-password

# Application Configuration
FLASK_SECRET_KEY=your-flask-secret-key
```

### Environment Variables Description

| Variable | Description | Required |
|----------|-------------|----------|
| `COSMOS_DB_URI` | Azure Cosmos DB account URI | Yes |
| `COSMOS_DB_KEY` | Azure Cosmos DB primary access key | Yes |
| `DATABASE_NAME` | Cosmos DB database name | Yes |
| `CONTAINER_NAME` | Cosmos DB container name for tickets | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI service API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint URL | Yes |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name for embedding model | Yes |
| `AZURE_OPENAI_GPT4O_DEPLOYMENT` | Deployment name for GPT-4 model | Yes |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | Yes |
| `MICROSOFT_APP_ID` | Microsoft Bot Framework App ID | For Teams |
| `MICROSOFT_APP_PASSWORD` | Microsoft Bot Framework App Password | For Teams |

## 🚀 Running the Application

### Local Development
```bash
python test.py
```

The application will be available at `http://localhost:5000`

### Using Docker
```bash
docker build -t autoresolvex .
docker run -p 5000:5000 autoresolvex
```

## 📚 API Endpoints

### Authentication
- `GET /login` - Login page
- `POST /login` - Authenticate user
- `GET /signup` - Registration page
- `POST /signup` - Register new user
- `GET /logout` - User logout

### Ticket Management
- `GET /tickets` - List all tickets
- `GET /ticket/<ticket_id>` - View specific ticket details
- `POST /ticket/<ticket_id>/ask` - AI assistant chat for specific ticket

### AI Services
- `POST /search-similar-incidents` - Find similar tickets using AI
- `POST /recommend-resolution` - Get AI-powered resolution recommendations
- `POST /api/ask-assistant` - General AI assistant endpoint

### Analytics
- `GET /analytics` - Analytics dashboard
- `GET /root-cause-patterns` - Root cause analysis and clustering

### Utilities
- `POST /generate-direct-link` - Generate secure access tokens
- `GET /direct-access/<token>` - Access tickets via secure token

### Bot Integration
- `POST /api/messages` - Microsoft Bot Framework webhook endpoint

## 🤖 Teams Bot Integration

### Bot Setup

1. **Create Bot in Azure Portal**
   - Navigate to Azure Bot Service
   - Create new bot resource
   - Configure messaging endpoint: `https://your-app-url.azurewebsites.net/api/messages`

2. **Configure Bot in Teams**
   - Use the provided `teamApp/manifest.json`
   - Update the manifest with your bot ID and app URL
   - Upload to Teams App Studio or Developer Portal

3. **Bot Commands**

| Command | Description | Example |
|---------|-------------|---------|
| `/similar [query]` | Find similar tickets | `/similar VPN connection issues` |
| `/recommend` | Get recommended tickets | `/recommend` |
| `/help` | Show help information | `/help` |

4. **Natural Language Queries**
The bot also supports natural language:
- "Find tickets about email problems"
- "Show me similar incidents for network issues"
- "What are the recommended solutions for printer problems?"

### Teams Bot Features
- **Contextual Responses**: AI-powered responses specific to ticket context
- **Markdown Support**: Rich text formatting in Teams messages
- **Adaptive Cards**: Interactive elements for better user experience
- **Direct Links**: Generate shareable links within Teams conversations

## 📖 Usage Guide

### Getting Started

1. **Login**: Access the application at the live URL and login with credentials
2. **Dashboard**: View ticket overview and analytics
3. **Ticket Search**: Use the search functionality to find specific tickets
4. **AI Assistant**: Interact with the AI for help and recommendations

### Using the AI Assistant

The AI assistant provides:
- **Contextual Help**: Ticket-specific guidance and solutions
- **Similar Tickets**: Find related incidents and their resolutions
- **Best Practices**: Recommendations based on historical data
- **Troubleshooting**: Step-by-step resolution guidance

### Analytics Features

- **Cluster Analysis**: View automatically identified ticket patterns
- **Root Cause Analysis**: AI-generated insights into systemic issues
- **Performance Metrics**: Track resolution times and success rates
- **Temporal Analysis**: Understand incident trends over time

### Direct Access Feature

Generate secure links for:
- **Sharing tickets** in Teams or email
- **Bypassing authentication** for 24-hour access
- **Emergency access** during critical incidents

## 🚀 Azure Deployment

### App Service Deployment

1. **Create App Service**
```bash
az webapp create --resource-group <rg-name> --plan <plan-name> --name <app-name> --runtime "PYTHON|3.11"
```

2. **Configure Application Settings**
```bash
az webapp config appsettings set --resource-group <rg-name> --name <app-name> --settings @appsettings.json
```

3. **Deploy Code**
```bash
az webapp deployment source config --resource-group <rg-name> --name <app-name> --repo-url <git-url> --branch main
```

### Container Deployment

1. **Build and Push Container**
```bash
docker build -t autoresolvex .
docker tag autoresolvex <registry-url>/autoresolvex:latest
docker push <registry-url>/autoresolvex:latest
```

2. **Deploy to App Service**
```bash
az webapp create --resource-group <rg-name> --plan <plan-name> --name <app-name> --deployment-container-image-name <registry-url>/autoresolvex:latest
```

### Environment Variables for Production

Set the following in Azure App Service Configuration:

```
COSMOS_DB_URI=<production-cosmos-uri>
COSMOS_DB_KEY=<production-cosmos-key>
AZURE_OPENAI_API_KEY=<production-openai-key>
AZURE_OPENAI_ENDPOINT=<production-openai-endpoint>
```

## 🔧 Configuration

### Performance Tuning
```python
# In test.py - adjust these values based on your needs
COSMOS_FETCH_LIMIT_FOR_SIMILARITY = 300
SIMILARITY_THRESHOLD = 0.75
RECOMMENDATION_THRESHOLD = 0.70
EMBEDDING_CACHE_SIZE = 1000
MAX_WORKERS = 4
```

### Security Settings
```python
# Flask security configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
```

## 🔒 Security Features

- **Token-based Authentication**: Secure JWT tokens for direct access
- **Session Management**: Secure session handling with HTTPOnly cookies
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting to prevent abuse
- **HTTPS Enforcement**: SSL/TLS encryption for all communications

## 📊 Monitoring & Logging

The application includes comprehensive logging:
- **Request Logging**: All API requests are logged
- **Error Tracking**: Detailed error logs with stack traces
- **Performance Metrics**: Response times and resource usage
- **AI Usage**: OpenAI API usage and costs tracking

### Log Files
- Application logs: Available in Azure App Service logs
- Bot logs: Microsoft Bot Framework Analytics
- AI logs: Azure OpenAI usage metrics

## 📈 Performance Optimization

- **Caching**: Redis caching for frequently accessed data
- **Connection Pooling**: Optimized database connections
- **Async Processing**: Background tasks for heavy operations
- **CDN**: Content delivery network for static assets


**AutoResolveX** - Transforming IT Support with AI
