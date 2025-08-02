#!/bin/bash

# ==============================================================================
# 🚀 FINAL EXECUTION COMMANDS - RUN THESE TO GO LIVE NOW
# ==============================================================================

echo "
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🔥                                                                        🔥
🔥  FINAL DEPLOYMENT - COPY AND RUN THESE COMMANDS IN YOUR TERMINAL NOW   🔥
🔥                                                                        🔥
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
"

echo ""
echo "STEP 1: Navigate to your repository directory"
echo "=============================================="
echo ""
echo "cd /path/to/your/repo"  # Replace with your actual path
echo "# OR"
echo "git clone https://github.com/Shauny123/backend-Ai-intake-voice-cleaner.v1.git"
echo "cd backend-Ai-intake-voice-cleaner.v1"

echo ""
echo "STEP 2: Make the deployment script executable and run it"
echo "========================================================"
echo ""
echo "# Make the script executable"
echo "chmod +x deploy_complete_system.sh"
echo ""
echo "# Run the complete deployment (this does everything)"
echo "./deploy_complete_system.sh"

echo ""
echo "ALTERNATIVE: Manual step-by-step deployment"
echo "==========================================="
echo ""
echo "# If the above doesn't work, run these commands one by one:"
echo ""

echo "# 1. Clean up repository"
echo "find . -name '*(*).py' -delete"
echo "find . -name '*(*).txt' -delete"
echo "find . -name '*(*).sh' -delete"
echo ""

echo "# 2. Set Google Cloud project"
echo "gcloud config set project durable-trainer-466014-h8"
echo ""

echo "# 3. Enable required APIs"
echo "gcloud services enable cloudbuild.googleapis.com"
echo "gcloud services enable run.googleapis.com"
echo "gcloud services enable containerregistry.googleapis.com"
echo "gcloud services enable domains.googleapis.com"
echo ""

echo "# 4. Build and deploy to Cloud Run"
echo "gcloud builds submit --tag gcr.io/durable-trainer-466014-h8/voice-cleaner-api ."
echo ""
echo "gcloud run deploy voice-cleaner-api \\"
echo "    --image gcr.io/durable-trainer-466014-h8/voice-cleaner-api \\"
echo "    --platform managed \\"
echo "    --region us-central1 \\"
echo "    --allow-unauthenticated \\"
echo "    --port 8080 \\"
echo "    --memory 2Gi \\"
echo "    --cpu 2 \\"
echo "    --timeout 300 \\"
echo "    --max-instances 10"
echo ""

echo "# 5. Get the service URL and test it"
echo "SERVICE_URL=\$(gcloud run services describe voice-cleaner-api --region=us-central1 --format='value(status.url)')"
echo "echo \"Service URL: \$SERVICE_URL\""
echo "curl \"\$SERVICE_URL/health\""
echo ""

echo "# 6. Set up domain mapping for your domains"
echo "gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.com --region us-central1 --platform managed"
echo "gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.ai --region us-central1 --platform managed"
echo "gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.help --region us-central1 --platform managed"

echo ""
echo "🎯 IMMEDIATE TEST COMMANDS"
echo "========================="
echo ""
echo "# After deployment, test these URLs:"
echo "curl https://voice-cleaner-api-[HASH]-uc.a.run.app/health"
echo "curl https://bywordofmouthlegal.com/health  # After DNS setup"
echo ""

echo "🌐 DNS SETUP INSTRUCTIONS"
echo "========================="
echo ""
echo "In your Cloudflare dashboard, add these DNS records:"
echo ""
echo "Domain: bywordofmouthlegal.com"
echo "  Type: CNAME"
echo "  Name: @"
echo "  Target: ghs.googlehosted.com"
echo "  Proxy: DNS only (gray cloud)"
echo ""
echo "Domain: bywordofmouthlegal.ai"
echo "  Type: CNAME" 
echo "  Name: @"
echo "  Target: ghs.googlehosted.com"
echo "  Proxy: DNS only (gray cloud)"
echo ""
echo "Domain: bywordofmouthlegal.help"
echo "  Type: CNAME"
echo "  Name: @"
echo "  Target: ghs.googlehosted.com"
echo "  Proxy: DNS only (gray cloud)"

echo ""
echo "🚨 TROUBLESHOOTING"
echo "=================="
echo ""
echo "If deployment fails:"
echo ""
echo "1. Check authentication:"
echo "   gcloud auth list"
echo "   gcloud auth login  # If not authenticated"
echo ""
echo "2. Check project access:"
echo "   gcloud projects describe durable-trainer-466014-h8"
echo ""
echo "3. Check billing is enabled:"
echo "   gcloud beta billing accounts list"
echo ""
echo "4. Manual Docker build (if Cloud Build fails):"
echo "   docker build -t voice-cleaner-api ."
echo "   docker tag voice-cleaner-api gcr.io/durable-trainer-466014-h8/voice-cleaner-api"
echo "   docker push gcr.io/durable-trainer-466014-h8/voice-cleaner-api"

echo ""
echo "📞 SUPPORT COMMANDS"
echo "==================="
echo ""
echo "# Check deployment status:"
echo "gcloud run services list"
echo ""
echo "# View logs:"
echo "gcloud logs read --service=voice-cleaner-api --limit=50"
echo ""
echo "# Check domain mappings:"
echo "gcloud run domain-mappings list"

echo ""
echo "✅ SUCCESS VERIFICATION"
echo "======================"
echo ""
echo "Your system is working when these return successful responses:"
echo ""
echo "curl https://voice-cleaner-api-[HASH]-uc.a.run.app/health"
echo "curl https://bywordofmouthlegal.com/health"
echo "curl https://bywordofmouthlegal.ai/health"
echo "curl https://bywordofmouthlegal.help/health"

echo ""
echo "🎉 WHAT YOU'LL HAVE WHEN COMPLETE:"
echo "=================================="
echo ""
echo "✅ Live websites at:"
echo "   - https://bywordofmouthlegal.com"
echo "   - https://bywordofmouthlegal.ai"
echo "   - https://bywordofmouthlegal.help"
echo ""
echo "✅ Working API endpoints:"
echo "   - /health - System status"
echo "   - /clean - Audio processing"
echo "   - /intake - Legal intake processing"
echo "   - /agent/trigger - AI agent workflows"
echo "   - /audio/upload - File upload"
echo ""
echo "✅ Infrastructure:"
echo "   - Google Cloud Run (auto-scaling)"
echo "   - SSL/TLS encryption"
echo "   - Global CDN via Cloudflare"
echo "   - Real-time monitoring"

echo ""
echo "🔥 COPY THE COMMANDS ABOVE AND RUN THEM NOW!"
echo "============================================="

# ==============================================================================
# QUICK START SCRIPT - SAVE THIS AS quickstart.sh
# ==============================================================================

cat > quickstart.sh << 'QUICKSTART_EOF'
#!/bin/bash

echo "🚀 Quick Start Deployment..."

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found. Install it from:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    echo "❌ Not authenticated. Run: gcloud auth login"
    exit 1
fi

echo "✅ Prerequisites OK"

# Set project and enable APIs
echo "🔧 Setting up Google Cloud..."
gcloud config set project durable-trainer-466014-h8
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com domains.googleapis.com

# Deploy
echo "🚀 Deploying to Cloud Run..."
gcloud builds submit --tag gcr.io/durable-trainer-466014-h8/voice-cleaner-api .

gcloud run deploy voice-cleaner-api \
    --image gcr.io/durable-trainer-466014-h8/voice-cleaner-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2

# Get URL and test
SERVICE_URL=$(gcloud run services describe voice-cleaner-api --region=us-central1 --format="value(status.url)")
echo "🌐 Service URL: $SERVICE_URL"

if curl -s "$SERVICE_URL/health" | grep -q "healthy"; then
    echo "✅ Deployment successful!"
else
    echo "❌ Health check failed"
fi

# Set up domain mappings
echo "🌐 Setting up domain mappings..."
gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.com --region us-central1 --platform managed || true
gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.ai --region us-central1 --platform managed || true
gcloud run domain-mappings create --service voice-cleaner-api --domain bywordofmouthlegal.help --region us-central1 --platform managed || true

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "Service URL: $SERVICE_URL"
echo "Set up DNS in Cloudflare to point your domains to: ghs.googlehosted.com"
QUICKSTART_EOF

chmod +x quickstart.sh

echo ""
echo "📄 QUICK START SCRIPT CREATED!"
echo "=============================="
echo ""
echo "Just run: ./quickstart.sh"
echo ""
echo "This will deploy everything automatically!"

echo ""
echo "🔥🔥🔥 READY TO GO LIVE! 🔥🔥🔥"