#!/bin/bash
# Deploy AI Agent to Enable All Required Google APIs
# This AI agent automatically enables 30+ APIs in parallel

PROJECT_ID="durable-trainer-466014-h8"

echo "🤖 Deploying AI API Enabler Agent..."
echo "📍 Project: $PROJECT_ID"
echo ""

# Create directory for AI agent
mkdir -p ~/ai-api-enabler
cd ~/ai-api-enabler

# Create the AI API Enabler script
cat > ai_api_enabler.py << 'EOF'
#!/usr/bin/env python3
"""
AI Agent: API Enabler - Automatically enables all required Google Cloud APIs
"""

import subprocess
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AIAPIEnabler:
    def __init__(self, project_id: str):
        self.project_id = project_id
        
        # All required APIs for AI orchestration
        self.required_apis = [
            # Core AI & Machine Learning
            "aiplatform.googleapis.com",
            "generativelanguage.googleapis.com", 
            "automl.googleapis.com",
            "speech.googleapis.com",
            "texttospeech.googleapis.com",
            "translate.googleapis.com",
            "documentai.googleapis.com",
            "videointelligence.googleapis.com",
            
            # Core Infrastructure
            "run.googleapis.com",
            "cloudbuild.googleapis.com",
            "functions.googleapis.com",
            "workflows.googleapis.com",
            "pubsub.googleapis.com",
            "secretmanager.googleapis.com",
            "eventarc.googleapis.com",
            
            # Monitoring & Operations
            "monitoring.googleapis.com",
            "logging.googleapis.com",
            "clouderrorreporting.googleapis.com",
            "cloudtrace.googleapis.com",
            "cloudprofiler.googleapis.com",
            
            # Security & Management
            "securitycenter.googleapis.com",
            "iam.googleapis.com",
            "cloudresourcemanager.googleapis.com",
            "serviceusage.googleapis.com",
            
            # Cost & Optimization
            "recommender.googleapis.com",
            "billing.googleapis.com",
            
            # Container & Storage
            "containerregistry.googleapis.com",
            "artifactregistry.googleapis.com",
            
            # Optional Enhancements
            "firebase.googleapis.com",
            "scheduler.googleapis.com"
        ]
        
        print(f"🤖 AI API Enabler initialized")
        print(f"📊 Total APIs to enable: {len(self.required_apis)}")

    def enable_api_fast(self, api_name: str):
        """Fast API enablement with error handling"""
        try:
            print(f"🔄 {api_name}")
            
            cmd = [
                "gcloud", "services", "enable", api_name,
                f"--project={self.project_id}",
                "--quiet"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"  ✅ {api_name}")
                return {"api": api_name, "success": True}
            else:
                print(f"  ⚠️ {api_name} - {result.stderr.strip()[:50]}")
                return {"api": api_name, "success": False, "error": result.stderr}
                
        except Exception as e:
            print(f"  ❌ {api_name} - {str(e)[:50]}")
            return {"api": api_name, 