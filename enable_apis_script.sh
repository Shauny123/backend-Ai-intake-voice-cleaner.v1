#!/bin/bash
# Enable missing APIs for AI Agent Orchestration

PROJECT_ID="durable-trainer-466014-h8"

echo "🤖 Enabling additional APIs for AI Agent Orchestration..."

# Core AI Platform APIs
echo "🧠 Enabling AI Platform APIs..."
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
gcloud services enable generativelanguage.googleapis.com --project=$PROJECT_ID

# Monitoring and Logging APIs
echo "📊 Enabling Monitoring APIs..."
gcloud services enable monitoring.googleapis.com --project=$PROJECT_ID
gcloud services enable logging.googleapis.com --project=$PROJECT_ID

# Container and Storage APIs
echo "🐳 Enabling Container APIs..."
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID

# Pub/Sub for agent communication
echo "📡 Enabling Pub/Sub API..."
gcloud services enable pubsub.googleapis.com --project=$PROJECT_ID

# IAM for service account management
echo "🔐 Enabling IAM API..."
gcloud services enable iam.googleapis.com --project=$PROJECT_ID

# Resource Manager for project operations
echo "📋 Enabling Resource Manager API..."
gcloud services enable cloudresourcemanager.googleapis.com --project=$PROJECT_ID

# Optional but recommended APIs
echo "⚡ Enabling optional enhancement APIs..."
gcloud services enable eventarc.googleapis.com --project=$PROJECT_ID  # For event-driven triggers
gcloud services enable firebase.googleapis.com --project=$PROJECT_ID   # For real-time notifications
gcloud services enable translate.googleapis.com --project=$PROJECT_ID  # For multilingual error analysis

echo "✅ All required APIs enabled!"
echo ""
echo "🎯 Your AI agents now have access to:"
echo "   • AI Platform (Gemini models for intelligent analysis)"
echo "   • Monitoring (performance tracking and alerts)"
echo "   • Container Registry (for storing optimized images)"
echo "   • Pub/Sub (for agent-to-agent communication)"
echo "   • Enhanced logging and error analysis"
echo ""
echo "🚀 Ready to deploy your AI orchestration system!"
