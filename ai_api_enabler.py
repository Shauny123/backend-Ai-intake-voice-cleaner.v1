#!/usr/bin/env python3
"""
AI Agent: API Enabler
Automatically enables all required Google Cloud APIs for the orchestration system
"""

import subprocess
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class AIAPIEnabler:
    def __init__(self, project_id: str):
        self.project_id = project_id
        
        # Complete list of APIs required for AI orchestration
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
            "scheduler.googleapis.com",
            "datacatalog.googleapis.com"
        ]
        
        print(f"🤖 AI API Enabler initialized for project: {project_id}")
        print(f"📊 Total APIs to manage: {len(self.required_apis)}")

    def enable_single_api(self, api_name: str) -> Dict[str, any]:
        """Enable a single API with intelligent retry logic"""
        try:
            print(f"🔄 Enabling {api_name}...")
            
            # Check if already enabled first
            check_cmd = [
                "gcloud", "services", "list", 
                "--enabled", 
                f"--filter=name:{api_name}",
                f"--project={self.project_id}",
                "--format=json"
            ]
            
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
            
            if check_result.returncode == 0:
                enabled_services = json.loads(check_result.stdout)
                if enabled_services:
                    print(f"  ✅ {api_name} already enabled")
                    return {"api": api_name, "status": "already_enabled", "success": True}
            
            # Enable the API
            enable_cmd = [
                "gcloud", "services", "enable", api_name,
                f"--project={self.project_id}",
                "--quiet"
            ]
            
            result = subprocess.run(enable_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"  ✅ {api_name} enabled successfully")
                return {"api": api_name, "status": "enabled", "success": True}
            else:
                error_msg = result.stderr.strip()
                print(f"  ⚠️ {api_name} - Issue: {error_msg}")
                
                # Intelligent retry for specific errors
                if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    print(f"  🔄 Quota issue detected for {api_name}, retrying in 10 seconds...")
                    time.sleep(10)
                    
                    retry_result = subprocess.run(enable_cmd, capture_output=True, text=True, timeout=120)
                    if retry_result.returncode == 0:
                        print(f"  ✅ {api_name} enabled on retry")
                        return {"api": api_name, "status": "enabled_on_retry", "success": True}
                
                return {"api": api_name, "status": "failed", "error": error_msg, "success": False}
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {api_name} - Timeout, will retry later")
            return {"api": api_name, "status": "timeout", "success": False}
        except Exception as e:
            print(f"  ❌ {api_name} - Error: {str(e)}")
            return {"api": api_name, "status": "error", "error": str(e), "success": False}

    async def enable_all_apis_parallel(self) -> Dict[str, any]:
        """Enable all APIs in parallel using async processing"""
        print("🚀 Starting parallel API enablement...")
        
        # Use ThreadPoolExecutor for parallel API enablement
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all API enablement tasks
            futures = {
                executor.submit(self.enable_single_api, api): api 
                for api in self.required_apis
            }
            
            results = []
            completed = 0
            total = len(self.required_apis)
            
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result(timeout=180)  # 3 minute timeout per API
                    results.append(result)
                    completed += 1
                    
                    progress = (completed / total) * 100
                    print(f"📊 Progress: {completed}/{total} ({progress:.1f}%)")
                    
                except Exception as e:
                    api_name = futures[future]
                    print(f"❌ Failed to enable {api_name}: {e}")
                    results.append({"api": api_name, "status": "failed", "error": str(e), "success": False})
        
        return self._analyze_results(results)

    def _analyze_results(self, results: List[Dict]) -> Dict[str, any]:
        """Analyze enablement results and provide summary"""
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        already_enabled = [r for r in successful if r.get("status") == "already_enabled"]
        newly_enabled = [r for r in successful if r.get("status") in ["enabled", "enabled_on_retry"]]
        
        summary = {
            "total_apis": len(self.required_apis),
            "successful": len(successful),
            "failed": len(failed),
            "already_enabled": len(already_enabled),
            "newly_enabled": len(newly_enabled),
            "success_rate": (len(successful) / len(self.required_apis)) * 100,
            "failed_apis": [r["api"] for r in failed],
            "results": results
        }
        
        return summary

    def retry_failed_apis(self, failed_apis: List[str]) -> Dict[str, any]:
        """Retry enabling failed APIs with enhanced error handling"""
        print(f"🔄 Retrying {len(failed_apis)} failed APIs...")
        
        retry_results = []
        for api in failed_apis:
            print(f"🔁 Retry attempt for {api}...")
            time.sleep(5)  # Wait between retries
            result = self.enable_single_api(api)
            retry_results.append(result)
        
        return self._analyze_results(retry_results)

    def verify_all_apis_enabled(self) -> Dict[str, any]:
        """Verify all required APIs are actually enabled"""
        print("🔍 Verifying all APIs are enabled...")
        
        try:
            cmd = [
                "gcloud", "services", "list", "--enabled",
                f"--project={self.project_id}",
                "--format=json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                enabled_services = json.loads(result.stdout)
                enabled_api_names = [service["name"] for service in enabled_services]
                
                missing_apis = []
                for required_api in self.required_apis:
                    if required_api not in enabled_api_names:
                        missing_apis.append(required_api)
                
                verification = {
                    "verification_successful": True,
                    "total_enabled": len(enabled_api_names),
                    "required_apis_enabled": len(self.required_apis) - len(missing_apis),
                    "missing_apis": missing_apis,
                    "completion_rate": ((len(self.required_apis) - len(missing_apis)) / len(self.required_apis)) * 100
                }
                
                if missing_apis:
                    print(f"⚠️ {len(missing_apis)} APIs still need to be enabled:")
                    for api in missing_apis:
                        print(f"   - {api}")
                else:
                    print("✅ All required APIs are enabled!")
                
                return verification
                
        except Exception as e:
            return {"verification_successful": False, "error": str(e)}

    def generate_api_status_report(self) -> str:
        """Generate a comprehensive API status report"""
        verification = self.verify_all_apis_enabled()
        
        report = f"""
🤖 AI API ENABLER - STATUS REPORT
=================================

📍 Project: {self.project_id}
📊 Total Required APIs: {len(self.required_apis)}
✅ APIs Enabled: {verification.get('required_apis_enabled', 'Unknown')}
⚠️ Missing APIs: {len(verification.get('missing_apis', []))}
📈 Completion Rate: {verification.get('completion_rate', 0):.1f}%

🔧 REQUIRED APIS STATUS:
"""
        
        if verification.get("verification_successful"):
            enabled_apis = set()
            try:
                cmd = ["gcloud", "services", "list", "--enabled", f"--project={self.project_id}", "--format=json"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    services = json.loads(result.stdout)
                    enabled_apis = {service["name"] for service in services}
            except:
                pass
            
            for api in self.required_apis:
                status = "✅ ENABLED" if api in enabled_apis else "❌ MISSING"
                report += f"   {status} - {api}\n"
                
            if verification.get("missing_apis"):
                report += f"\n🚨 MISSING APIS TO ENABLE:\n"
                for api in verification["missing_apis"]:
                    report += f"   gcloud services enable {api} --project={self.project_id}\n"
        
        report += "\n🎯 AI Agent API Enablement Complete!"
        
        return report

async def main():
    """Main execution function for AI API Enabler"""
    
    project_id = "durable-trainer-466014-h8"
    
    print("🤖 Starting AI-Powered API Enablement...")
    print(f"📍 Target Project: {project_id}")
    print("🎯 Mission: Enable all required APIs for AI orchestration")
    print()
    
    # Initialize AI API Enabler
    enabler = AIAPIEnabler(project_id)
    
    # Step 1: Enable all APIs in parallel
    print("🚀 Phase 1: Parallel API Enablement")
    results = await enabler.enable_all_apis_parallel()
    
    print(f"""
📊 PHASE 1 RESULTS:
   ✅ Successful: {results['successful']}/{results['total_apis']}
   ❌ Failed: {results['failed']}
   📈 Success Rate: {results['success_rate']:.1f}%
""")
    
    # Step 2: Retry failed APIs if any
    if results['failed'] > 0:
        print("🔄 Phase 2: Retrying Failed APIs")
        retry_results = enabler.retry_failed_apis(results['failed_apis'])
        print(f"🔁 Retry Results: {retry_results['successful']} additional APIs enabled")
    
    # Step 3: Final verification
    print("🔍 Phase 3: Final Verification")
    verification = enabler.verify_all_apis_enabled()
    
    # Step 4: Generate comprehensive report
    print("📋 Phase 4: Generating Status Report")
    report = enabler.generate_api_status_report()
    print(report)
    
    # Save report to file
    with open("api_enablement_report.txt", "w") as f:
        f.write(report)
    
    print("💾 Report saved to: api_enablement_report.txt")
    
    if verification.get("completion_rate", 0) >= 95:
        print("\n🎉 SUCCESS: AI API Enablement Complete!")
        print("🤖 All required APIs are now enabled for AI orchestration")
        return True
    else:
        print("\n⚠️ WARNING: Some APIs still need manual attention")
        return False

if __name__ == "__main__":
    # Run the AI API Enabler
    success = asyncio.run(main())
    exit(0 if success else 1)
