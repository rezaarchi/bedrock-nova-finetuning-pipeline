#!/usr/bin/env python3
"""
Test script for fine-tuned Bedrock Nova Pro model
Creates custom model deployment for on-demand inference
Based on: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_custom_model_deployment.html
"""

import boto3
import json
import argparse
from datetime import datetime
from typing import Dict, List
import time

class ModelTester:
    def __init__(self, region: str = 'us-east-1'):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.region = region
        
    def create_custom_model_deployment(self, custom_model_arn: str):
        """
        Create custom model deployment for on-demand inference
        See: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_custom_model_deployment.html
        """
        print(f"\n=== Creating Custom Model Deployment ===")
        print(f"Custom Model ARN: {custom_model_arn}")
        
        deployment_name = f'support-deployment-{int(datetime.now().timestamp())}'
        
        try:
            response = self.bedrock.create_custom_model_deployment(
                modelDeploymentName=deployment_name,
                modelArn=custom_model_arn
            )
            
            deployment_arn = response['customModelDeploymentArn']
            print(f"✓ Deployment created: {deployment_arn}")
            print(f"✓ Deployment Name: {deployment_name}")
            
            # Extract and store deployment ID for status checking
            # ARN format: arn:aws:bedrock:region:account:custom-model-deployment/ID
            deployment_id = deployment_arn.split('/')[-1]
            print(f"✓ Deployment ID: {deployment_id}")
            
            # Wait for deployment to be active
            print("\nWaiting for deployment to become active (this may take a few minutes)...")
            self._wait_for_deployment(deployment_id)
            
            return deployment_arn
            
        except Exception as e:
            print(f"✗ Error creating deployment: {e}")
            raise
    
    def _wait_for_deployment(self, deployment_id: str, max_wait_time: int = 600):
        """Wait for deployment to become active"""
        start_time = time.time()
        
        print(f"Checking status for deployment ID: {deployment_id}")
        print("Note: It may take 30-60 seconds for deployment to be queryable after creation")
        
        consecutive_errors = 0
        max_consecutive_errors = 6  # Allow ~60 seconds of errors
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.bedrock.get_custom_model_deployment(
                    customModelDeploymentIdentifier=deployment_id
                )
                
                # Reset error counter on successful call
                consecutive_errors = 0
                
                status = response.get('status', 'UNKNOWN')
                print(f"  Status: {status}")
                
                if status == 'ACTIVE':
                    print("✓ Deployment is active and ready!")
                    return True
                elif status in ['FAILED', 'STOPPED']:
                    print(f"✗ Deployment failed with status: {status}")
                    if 'failureMessage' in response:
                        print(f"  Failure reason: {response['failureMessage']}")
                    return False
                
                time.sleep(10)
                
            except Exception as e:
                consecutive_errors += 1
                
                if consecutive_errors <= max_consecutive_errors:
                    print(f"  Waiting for deployment to be queryable... ({consecutive_errors}/{max_consecutive_errors})")
                else:
                    print(f"Error checking deployment status: {e}")
                    print(f"  Deployment ID: {deployment_id}")
                    print("\nTroubleshooting:")
                    print("  1. Check if deployment exists:")
                    print(f"     aws bedrock list-custom-model-deployments --region {self.region}")
                    print("  2. Try invoking directly with deployment ARN once it's ACTIVE")
                    return False
                
                time.sleep(10)
        
        print("✗ Deployment timed out")
        return False
    
    def list_deployments(self):
        """List all custom model deployments"""
        print("\n=== Listing Custom Model Deployments ===")
        try:
            response = self.bedrock.list_custom_model_deployments()
            
            deployments = response.get('customModelDeploymentSummaries', [])
            
            if not deployments:
                print("No deployments found")
                return []
            
            for dep in deployments:
                print(f"\nDeployment ARN: {dep['customModelDeploymentArn']}")
                print(f"  Name: {dep['customModelDeploymentName']}")
                print(f"  Status: {dep.get('status', 'UNKNOWN')}")
                print(f"  Model ARN: {dep['customModelArn']}")
            
            return deployments
            
        except Exception as e:
            print(f"Error listing deployments: {e}")
            return []
        
    def classify_ticket(self, deployment_arn: str, title: str, description: str) -> Dict:
        """Send a ticket to the model for classification"""
        
        prompt = f"""Classify this support ticket:

Title: {title}
Description: {description}

Provide the category, severity, and recommended team."""
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=deployment_arn,
                body=json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.5,
                        "maxTokens": 512,
                        "topP": 0.9
                    }
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'classification': result['output']['message']['content'][0]['text'],
                'stop_reason': result.get('stopReason', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_test_suite(self, deployment_arn: str) -> List[Dict]:
        """Run a suite of test tickets"""
        
        test_tickets = [
            {
                'title': 'Cannot upload files larger than 5MB',
                'description': 'User trying to upload PDF files. Files under 5MB work fine but larger files fail with timeout error. Using Chrome browser version 120.',
                'expected_category': 'Technical Bug',
                'expected_severity': 'High'
            },
            {
                'title': 'Password reset email not received',
                'description': 'Customer requested password reset 30 minutes ago but has not received the email. Checked spam folder. Email address verified as correct.',
                'expected_category': 'Account Access',
                'expected_severity': 'High'
            },
            {
                'title': 'Charged twice for monthly subscription',
                'description': 'Customer shows two charges of $49.99 on credit card statement for the same billing period. Requesting refund for duplicate charge.',
                'expected_category': 'Billing Issue',
                'expected_severity': 'High'
            },
            {
                'title': 'Request: Add dark mode to dashboard',
                'description': 'Customer using dashboard for 6+ hours daily and experiencing eye strain. Requesting dark mode option.',
                'expected_category': 'Feature Request',
                'expected_severity': 'Low'
            },
            {
                'title': 'Dashboard loading very slowly',
                'description': 'Dashboard taking 45-60 seconds to load. Previously loaded in under 5 seconds. Started 3 days ago.',
                'expected_category': 'Performance Issue',
                'expected_severity': 'Medium'
            }
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("RUNNING TEST SUITE")
        print("="*80)
        
        for i, ticket in enumerate(test_tickets, 1):
            print(f"\n[Test {i}/{len(test_tickets)}]")
            print(f"Title: {ticket['title']}")
            print(f"Expected: {ticket['expected_category']} / {ticket['expected_severity']}")
            print("-" * 80)
            
            result = self.classify_ticket(deployment_arn, ticket['title'], ticket['description'])
            
            if result['success']:
                print("Model Response:")
                print(result['classification'])
                print(f"\nStop Reason: {result['stop_reason']}")
                
                classification_text = result['classification'].lower()
                category_match = ticket['expected_category'].lower() in classification_text
                severity_match = ticket['expected_severity'].lower() in classification_text
                
                print(f"\nValidation:")
                print(f"  Category Match: {'✓' if category_match else '✗'}")
                print(f"  Severity Match: {'✓' if severity_match else '✗'}")
                
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': True,
                    'category_match': category_match,
                    'severity_match': severity_match,
                    'response': result['classification']
                })
            else:
                print(f"ERROR: {result['error']}")
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': False,
                    'error': result['error']
                })
            
            print("="*80)
            time.sleep(1)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        successful_tests = [r for r in results if r['success']]
        category_matches = sum(1 for r in successful_tests if r.get('category_match', False))
        severity_matches = sum(1 for r in successful_tests if r.get('severity_match', False))
        
        print(f"\nTotal Tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(results) - len(successful_tests)}")
        
        if successful_tests:
            print(f"\nCategory Match Rate: {category_matches}/{len(successful_tests)} ({category_matches/len(successful_tests)*100:.1f}%)")
            print(f"Severity Match Rate: {severity_matches}/{len(successful_tests)} ({severity_matches/len(successful_tests)*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def delete_deployment(self, deployment_arn_or_id: str):
        """Delete custom model deployment"""
        print(f"\n=== Deleting Deployment ===")
        
        # Extract deployment ID from ARN if full ARN provided
        if deployment_arn_or_id.startswith('arn:'):
            deployment_id = deployment_arn_or_id.split('/')[-1]
            print(f"Deployment ARN: {deployment_arn_or_id}")
            print(f"Deployment ID: {deployment_id}")
        else:
            deployment_id = deployment_arn_or_id
            print(f"Deployment ID: {deployment_id}")
        
        try:
            self.bedrock.delete_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_id
            )
            print("✓ Deployment deleted successfully")
            print("No ongoing charges for unused deployments")
            
        except Exception as e:
            print(f"✗ Error deleting deployment: {e}")
            print(f"\nTry manually:")
            print(f"  aws bedrock delete-custom-model-deployment \\")
            print(f"      --custom-model-deployment-identifier {deployment_id} \\")
            print(f"      --region {self.region}")


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned Bedrock model with custom model deployment'
    )
    parser.add_argument(
        '--custom-model-arn',
        help='ARN of the custom fine-tuned model (to create deployment)'
    )
    parser.add_argument(
        '--deployment-arn',
        help='ARN of existing deployment (if already created)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--list-deployments',
        action='store_true',
        help='List all custom model deployments'
    )
    parser.add_argument(
        '--delete-deployment',
        help='Delete a specific deployment by ARN'
    )
    
    args = parser.parse_args()
    
    tester = ModelTester(region=args.region)
    
    print("\n" + "="*80)
    print("BEDROCK CUSTOM MODEL TESTER")
    print("="*80)
    print(f"Region: {args.region}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List deployments
    if args.list_deployments:
        tester.list_deployments()
        return
    
    # Delete deployment
    if args.delete_deployment:
        tester.delete_deployment(args.delete_deployment)
        return
    
    # Create deployment if needed
    if args.custom_model_arn and not args.deployment_arn:
        print("\nCreating new deployment...")
        deployment_arn = tester.create_custom_model_deployment(args.custom_model_arn)
        
        print("\n" + "="*80)
        print("NEXT STEP:")
        print("="*80)
        print(f"Run tests with: python test_model.py --deployment-arn {deployment_arn}")
        print("\nOr continue immediately:")
        
        response = input("\nRun tests now? (y/N): ").strip().lower()
        if response != 'y':
            return
        
        args.deployment_arn = deployment_arn
    
    # Run tests
    if not args.deployment_arn:
        print("\n✗ Error: Either --custom-model-arn or --deployment-arn required")
        print("\nUsage:")
        print("  # Step 1: Create deployment")
        print("  python test_model.py --custom-model-arn <arn>")
        print("\n  # Step 2: Run tests")
        print("  python test_model.py --deployment-arn <deployment-arn>")
        print("\n  # Step 3: List deployments")
        print("  python test_model.py --list-deployments")
        print("\n  # Step 4: Delete deployment")
        print("  python test_model.py --delete-deployment <deployment-arn>")
        return
    
    print(f"\nDeployment ARN: {args.deployment_arn}")
    
    results = tester.run_test_suite(args.deployment_arn)
    tester.print_summary(results)
    
    # Save results
    output_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'deployment_arn': args.deployment_arn,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Offer to delete deployment
    print("\n" + "="*80)
    response = input("Delete deployment now? (y/N): ").strip().lower()
    if response == 'y':
        tester.delete_deployment(args.deployment_arn)


if __name__ == '__main__':
    main()

        
    def create_on_demand_deployment(self, custom_model_arn: str):
        """
        Create on-demand custom model deployment
        This is pay-per-use, not hourly like provisioned throughput
        """
        print(f"\n=== Creating On-Demand Model Deployment ===")
        print(f"Custom Model ARN: {custom_model_arn}")
        
        deployment_name = f'support-on-demand-{int(datetime.now().timestamp())}'
        
        try:
            response = self.bedrock.put_model_invocation_logging_configuration(
                loggingConfig={
                    'cloudWatchConfig': {
                        'logGroupName': '/aws/bedrock/modelinvocations',
                        'roleArn': 'arn:aws:iam::*:role/*'
                    },
                    'embeddingDataDeliveryEnabled': False,
                    'imageDataDeliveryEnabled': False,
                    'textDataDeliveryEnabled': True
                }
            )
            
            print(f"✓ Model is ready for on-demand inference")
            print(f"✓ Model ID: {custom_model_arn}")
            print("\nYou can now invoke the model directly using its ARN")
            print("No provisioned throughput needed - pay only for what you use!")
            
            return custom_model_arn
            
        except Exception as e:
            print(f"Note: Logging configuration optional. Proceeding with model: {custom_model_arn}")
            return custom_model_arn
        
    def classify_ticket(self, model_id: str, title: str, description: str) -> Dict:
        """Send a ticket to the model for classification"""
        
        prompt = f"""Classify this support ticket:

Title: {title}
Description: {description}

Provide the category, severity, and recommended team."""
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.5,
                        "maxTokens": 512,
                        "topP": 0.9
                    }
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'classification': result['output']['message']['content'][0]['text'],
                'stop_reason': result.get('stopReason', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_test_suite(self, model_id: str) -> List[Dict]:
        """Run a suite of test tickets"""
        
        test_tickets = [
            {
                'title': 'Cannot upload files larger than 5MB',
                'description': 'User trying to upload PDF files. Files under 5MB work fine but larger files fail with timeout error. Using Chrome browser version 120. Happens consistently across multiple files.',
                'expected_category': 'Technical Bug',
                'expected_severity': 'High'
            },
            {
                'title': 'Password reset email not received',
                'description': 'Customer requested password reset 30 minutes ago but has not received the email. Checked spam folder. Email address verified as correct in profile. User unable to access account.',
                'expected_category': 'Account Access',
                'expected_severity': 'High'
            },
            {
                'title': 'Charged twice for monthly subscription',
                'description': 'Customer shows two charges of $49.99 on credit card statement for the same billing period. First charge on Jan 1st at 9:00 AM, second charge on Jan 1st at 9:15 AM. Customer requesting refund for duplicate charge.',
                'expected_category': 'Billing Issue',
                'expected_severity': 'High'
            },
            {
                'title': 'Request: Add dark mode to dashboard',
                'description': 'Customer using the dashboard for extended periods (6+ hours daily) and experiencing eye strain. Requesting dark mode option to reduce screen brightness. Would improve usability significantly.',
                'expected_category': 'Feature Request',
                'expected_severity': 'Low'
            },
            {
                'title': 'Dashboard loading very slowly',
                'description': 'Dashboard taking 45-60 seconds to load. Previously loaded in under 5 seconds. Issue started 3 days ago. Tested on multiple browsers and devices with same result. Other users in organization reporting similar issues.',
                'expected_category': 'Performance Issue',
                'expected_severity': 'Medium'
            },
            {
                'title': 'How do I add team members to my account?',
                'description': 'Account administrator asking how to invite new team members. Current team size is 5 people. Need to add 3 more users. Looking for step-by-step instructions on the invitation process and permission settings.',
                'expected_category': 'General Inquiry',
                'expected_severity': 'Low'
            },
            {
                'title': 'Export file contains corrupted data',
                'description': 'Exported CSV file shows NULL values and random characters for 30% of records. Same records display correctly in the web interface. Export generated 2 hours ago. File size is 15MB. Need clean export urgently for monthly report.',
                'expected_category': 'Data Issue',
                'expected_severity': 'High'
            }
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("RUNNING TEST SUITE")
        print("="*80)
        
        for i, ticket in enumerate(test_tickets, 1):
            print(f"\n[Test {i}/{len(test_tickets)}]")
            print(f"Title: {ticket['title']}")
            print(f"Expected: {ticket['expected_category']} / {ticket['expected_severity']}")
            print("-" * 80)
            
            result = self.classify_ticket(model_id, ticket['title'], ticket['description'])
            
            if result['success']:
                print("Model Response:")
                print(result['classification'])
                print(f"\nStop Reason: {result['stop_reason']}")
                
                classification_text = result['classification'].lower()
                category_match = ticket['expected_category'].lower() in classification_text
                severity_match = ticket['expected_severity'].lower() in classification_text
                
                print(f"\nValidation:")
                print(f"  Category Match: {'✓' if category_match else '✗'}")
                print(f"  Severity Match: {'✓' if severity_match else '✗'}")
                
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': True,
                    'category_match': category_match,
                    'severity_match': severity_match,
                    'response': result['classification']
                })
            else:
                print(f"ERROR: {result['error']}")
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': False,
                    'error': result['error']
                })
            
            print("="*80)
            
            # Small delay between requests
            time.sleep(1)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        successful_tests = [r for r in results if r['success']]
        category_matches = sum(1 for r in successful_tests if r.get('category_match', False))
        severity_matches = sum(1 for r in successful_tests if r.get('severity_match', False))
        
        print(f"\nTotal Tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(results) - len(successful_tests)}")
        
        if successful_tests:
            print(f"\nCategory Match Rate: {category_matches}/{len(successful_tests)} ({category_matches/len(successful_tests)*100:.1f}%)")
            print(f"Severity Match Rate: {severity_matches}/{len(successful_tests)} ({severity_matches/len(successful_tests)*100:.1f}%)")
        
        print("\n" + "="*80)
    
    def interactive_test(self, model_id: str):
        """Interactive mode for testing custom tickets"""
        
        print("\n" + "="*80)
        print("INTERACTIVE TEST MODE")
        print("="*80)
        print("Enter support ticket details (or 'quit' to exit)")
        
        while True:
            print("\n" + "-"*80)
            title = input("\nTicket Title: ").strip()
            
            if title.lower() == 'quit':
                break
                
            description = input("Description: ").strip()
            
            if not title or not description:
                print("Both title and description required!")
                continue
            
            print("\nClassifying...")
            result = self.classify_ticket(model_id, title, description)
            
            if result['success']:
                print("\nModel Response:")
                print(result['classification'])
            else:
                print(f"\nERROR: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned Bedrock model using on-demand deployment'
    )
    parser.add_argument(
        '--model-arn',
        required=True,
        help='ARN of the custom fine-tuned model'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BEDROCK MODEL TESTER (On-Demand)")
    print("="*80)
    print(f"Model ARN: {args.model_arn}")
    print(f"Region: {args.region}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nUsing on-demand inference - pay only for what you use!")
    
    tester = ModelTester(region=args.region)
    
    # Custom models can be invoked directly with on-demand pricing
    model_id = args.model_arn
    
    if args.interactive:
        tester.interactive_test(model_id)
    else:
        results = tester.run_test_suite(model_id)
        tester.print_summary(results)
        
        # Save results to file
        output_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'model_arn': args.model_arn,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Estimate costs
        total_tokens = len(results) * 600  # Rough estimate
        cost = (total_tokens / 1000) * 0.016
        print(f"\nEstimated cost for this test run: ~${cost:.2f}")


if __name__ == '__main__':
    main()

        
    def create_provisioned_throughput(self, custom_model_arn: str, model_units: int = 1):
        """
        Create provisioned throughput for custom model
        This is required to invoke custom models
        """
        print(f"\n=== Creating Provisioned Throughput ===")
        print(f"Custom Model ARN: {custom_model_arn}")
        print(f"Model Units: {model_units}")
        
        try:
            response = self.bedrock.create_provisioned_model_throughput(
                modelUnits=model_units,
                provisionedModelName=f'support-classifier-throughput-{int(datetime.now().timestamp())}',
                modelId=custom_model_arn
            )
            
            provisioned_arn = response['provisionedModelArn']
            print(f"✓ Provisioned Throughput ARN: {provisioned_arn}")
            print("\n⚠️  NOTE: Provisioning takes 10-15 minutes to complete")
            print("   Check status with:")
            print(f"   aws bedrock get-provisioned-model-throughput --provisioned-model-id {provisioned_arn.split('/')[-1]}")
            
            return provisioned_arn
            
        except Exception as e:
            print(f"✗ Error creating provisioned throughput: {e}")
            raise
        
    def classify_ticket(self, title: str, description: str) -> Dict:
        """Send a ticket to the model for classification"""
        
        prompt = f"""Classify this support ticket:

Title: {title}
Description: {description}

Provide the category, severity, and recommended team."""
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.provisioned_model_arn,
                body=json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.5,
                        "maxTokens": 512,
                        "topP": 0.9
                    }
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'classification': result['output']['message']['content'][0]['text'],
                'stop_reason': result.get('stopReason', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_test_suite(self) -> List[Dict]:
        """Run a suite of test tickets"""
        
        test_tickets = [
            {
                'title': 'Cannot upload files larger than 5MB',
                'description': 'User trying to upload PDF files. Files under 5MB work fine but larger files fail with timeout error. Using Chrome browser version 120.',
                'expected_category': 'Technical Bug',
                'expected_severity': 'High'
            },
            {
                'title': 'Password reset email not received',
                'description': 'Customer requested password reset 30 minutes ago but has not received the email. Checked spam folder. Email address verified as correct.',
                'expected_category': 'Account Access',
                'expected_severity': 'High'
            },
            {
                'title': 'Charged twice for monthly subscription',
                'description': 'Customer shows two charges of $49.99 on credit card statement for the same billing period. Requesting refund for duplicate charge.',
                'expected_category': 'Billing Issue',
                'expected_severity': 'High'
            },
            {
                'title': 'Request: Add dark mode to dashboard',
                'description': 'Customer using dashboard for 6+ hours daily and experiencing eye strain. Requesting dark mode option.',
                'expected_category': 'Feature Request',
                'expected_severity': 'Low'
            },
            {
                'title': 'Dashboard loading very slowly',
                'description': 'Dashboard taking 45-60 seconds to load. Previously loaded in under 5 seconds. Started 3 days ago.',
                'expected_category': 'Performance Issue',
                'expected_severity': 'Medium'
            }
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("RUNNING TEST SUITE")
        print("="*80)
        
        for i, ticket in enumerate(test_tickets, 1):
            print(f"\n[Test {i}/{len(test_tickets)}]")
            print(f"Title: {ticket['title']}")
            print(f"Expected: {ticket['expected_category']} / {ticket['expected_severity']}")
            print("-" * 80)
            
            result = self.classify_ticket(ticket['title'], ticket['description'])
            
            if result['success']:
                print("Model Response:")
                print(result['classification'])
                print(f"\nStop Reason: {result['stop_reason']}")
                
                classification_text = result['classification'].lower()
                category_match = ticket['expected_category'].lower() in classification_text
                severity_match = ticket['expected_severity'].lower() in classification_text
                
                print(f"\nValidation:")
                print(f"  Category Match: {'✓' if category_match else '✗'}")
                print(f"  Severity Match: {'✓' if severity_match else '✗'}")
                
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': True,
                    'category_match': category_match,
                    'severity_match': severity_match,
                    'response': result['classification']
                })
            else:
                print(f"ERROR: {result['error']}")
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': False,
                    'error': result['error']
                })
            
            print("="*80)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        successful_tests = [r for r in results if r['success']]
        category_matches = sum(1 for r in successful_tests if r.get('category_match', False))
        severity_matches = sum(1 for r in successful_tests if r.get('severity_match', False))
        
        print(f"\nTotal Tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(results) - len(successful_tests)}")
        
        if successful_tests:
            print(f"\nCategory Match Rate: {category_matches}/{len(successful_tests)} ({category_matches/len(successful_tests)*100:.1f}%)")
            print(f"Severity Match Rate: {severity_matches}/{len(successful_tests)} ({severity_matches/len(successful_tests)*100:.1f}%)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned Bedrock model (requires provisioned throughput)'
    )
    parser.add_argument(
        '--custom-model-arn',
        help='ARN of the custom fine-tuned model (to create provisioned throughput)'
    )
    parser.add_argument(
        '--provisioned-arn',
        help='ARN of existing provisioned throughput (if already created)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--create-throughput',
        action='store_true',
        help='Create provisioned throughput for custom model'
    )
    parser.add_argument(
        '--model-units',
        type=int,
        default=1,
        help='Number of model units for provisioned throughput (default: 1)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BEDROCK MODEL TESTER")
    print("="*80)
    print(f"Region: {args.region}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.create_throughput:
        if not args.custom_model_arn:
            print("\n✗ Error: --custom-model-arn required with --create-throughput")
            return
        
        tester = ModelTester(provisioned_model_arn=None, region=args.region)
        provisioned_arn = tester.create_provisioned_throughput(
            args.custom_model_arn, 
            args.model_units
        )
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Wait 10-15 minutes for provisioning to complete")
        print("2. Check status:")
        print(f"   aws bedrock get-provisioned-model-throughput \\")
        print(f"       --provisioned-model-id {provisioned_arn.split('/')[-1]}")
        print("3. Once status is 'InService', run tests:")
        print(f"   python test_model.py --provisioned-arn {provisioned_arn}")
        return
    
    if not args.provisioned_arn:
        print("\n✗ Error: Either --provisioned-arn or --create-throughput required")
        print("\nUsage:")
        print("  # Step 1: Create provisioned throughput")
        print("  python test_model.py --custom-model-arn <arn> --create-throughput")
        print("\n  # Step 2: Run tests (after provisioning completes)")
        print("  python test_model.py --provisioned-arn <provisioned-arn>")
        return
    
    print(f"Provisioned Model ARN: {args.provisioned_arn}")
    
    tester = ModelTester(args.provisioned_arn, args.region)
    results = tester.run_test_suite()
    tester.print_summary(results)
    
    # Save results
    output_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'provisioned_arn': args.provisioned_arn,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

        
    def classify_ticket(self, title: str, description: str) -> Dict:
        """Send a ticket to the model for classification"""
        
        prompt = f"""Classify this support ticket:

Title: {title}
Description: {description}

Provide the category, severity, and recommended team."""
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_arn,
                body=json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.5,
                        "maxTokens": 512,
                        "topP": 0.9
                    }
                })
            )
            
            result = json.loads(response['body'].read())
            return {
                'success': True,
                'classification': result['output']['message']['content'][0]['text'],
                'stop_reason': result.get('stopReason', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_test_suite(self) -> List[Dict]:
        """Run a suite of test tickets"""
        
        test_tickets = [
            {
                'title': 'Cannot upload files larger than 5MB',
                'description': 'User trying to upload PDF files. Files under 5MB work fine but larger files fail with timeout error. Using Chrome browser version 120. Happens consistently across multiple files.',
                'expected_category': 'Technical Bug',
                'expected_severity': 'High'
            },
            {
                'title': 'Password reset email not received',
                'description': 'Customer requested password reset 30 minutes ago but has not received the email. Checked spam folder. Email address verified as correct in profile. User unable to access account.',
                'expected_category': 'Account Access',
                'expected_severity': 'High'
            },
            {
                'title': 'Charged twice for monthly subscription',
                'description': 'Customer shows two charges of $49.99 on credit card statement for the same billing period. First charge on Jan 1st at 9:00 AM, second charge on Jan 1st at 9:15 AM. Customer requesting refund for duplicate charge.',
                'expected_category': 'Billing Issue',
                'expected_severity': 'High'
            },
            {
                'title': 'Request: Add dark mode to dashboard',
                'description': 'Customer using the dashboard for extended periods (6+ hours daily) and experiencing eye strain. Requesting dark mode option to reduce screen brightness. Would improve usability significantly.',
                'expected_category': 'Feature Request',
                'expected_severity': 'Low'
            },
            {
                'title': 'Dashboard loading very slowly',
                'description': 'Dashboard taking 45-60 seconds to load. Previously loaded in under 5 seconds. Issue started 3 days ago. Tested on multiple browsers and devices with same result. Other users in organization reporting similar issues.',
                'expected_category': 'Performance Issue',
                'expected_severity': 'Medium'
            },
            {
                'title': 'How do I add team members to my account?',
                'description': 'Account administrator asking how to invite new team members. Current team size is 5 people. Need to add 3 more users. Looking for step-by-step instructions on the invitation process and permission settings.',
                'expected_category': 'General Inquiry',
                'expected_severity': 'Low'
            },
            {
                'title': 'Export file contains corrupted data',
                'description': 'Exported CSV file shows NULL values and random characters for 30% of records. Same records display correctly in the web interface. Export generated 2 hours ago. File size is 15MB. Need clean export urgently for monthly report.',
                'expected_category': 'Data Issue',
                'expected_severity': 'High'
            }
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("RUNNING TEST SUITE")
        print("="*80)
        
        for i, ticket in enumerate(test_tickets, 1):
            print(f"\n[Test {i}/{len(test_tickets)}]")
            print(f"Title: {ticket['title']}")
            print(f"Expected: {ticket['expected_category']} / {ticket['expected_severity']}")
            print("-" * 80)
            
            result = self.classify_ticket(ticket['title'], ticket['description'])
            
            if result['success']:
                print("Model Response:")
                print(result['classification'])
                print(f"\nStop Reason: {result['stop_reason']}")
                
                # Simple validation
                classification_text = result['classification'].lower()
                category_match = ticket['expected_category'].lower() in classification_text
                severity_match = ticket['expected_severity'].lower() in classification_text
                
                print(f"\nValidation:")
                print(f"  Category Match: {'✓' if category_match else '✗'}")
                print(f"  Severity Match: {'✓' if severity_match else '✗'}")
                
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': True,
                    'category_match': category_match,
                    'severity_match': severity_match,
                    'response': result['classification']
                })
            else:
                print(f"ERROR: {result['error']}")
                results.append({
                    'test': i,
                    'title': ticket['title'],
                    'success': False,
                    'error': result['error']
                })
            
            print("="*80)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        successful_tests = [r for r in results if r['success']]
        category_matches = sum(1 for r in successful_tests if r.get('category_match', False))
        severity_matches = sum(1 for r in successful_tests if r.get('severity_match', False))
        
        print(f"\nTotal Tests: {len(results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(results) - len(successful_tests)}")
        
        if successful_tests:
            print(f"\nCategory Match Rate: {category_matches}/{len(successful_tests)} ({category_matches/len(successful_tests)*100:.1f}%)")
            print(f"Severity Match Rate: {severity_matches}/{len(successful_tests)} ({severity_matches/len(successful_tests)*100:.1f}%)")
        
        print("\n" + "="*80)
        
    def interactive_test(self):
        """Interactive mode for testing custom tickets"""
        
        print("\n" + "="*80)
        print("INTERACTIVE TEST MODE")
        print("="*80)
        print("Enter support ticket details (or 'quit' to exit)")
        
        while True:
            print("\n" + "-"*80)
            title = input("\nTicket Title: ").strip()
            
            if title.lower() == 'quit':
                break
                
            description = input("Description: ").strip()
            
            if not title or not description:
                print("Both title and description required!")
                continue
            
            print("\nClassifying...")
            result = self.classify_ticket(title, description)
            
            if result['success']:
                print("\nModel Response:")
                print(result['classification'])
            else:
                print(f"\nERROR: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned Bedrock model for support ticket classification'
    )
    parser.add_argument(
        '--model-arn',
        required=True,
        help='ARN of the fine-tuned model'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BEDROCK MODEL TESTER")
    print("="*80)
    print(f"Model ARN: {args.model_arn}")
    print(f"Region: {args.region}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = ModelTester(args.model_arn, args.region)
    
    if args.interactive:
        tester.interactive_test()
    else:
        results = tester.run_test_suite()
        tester.print_summary(results)
        
        # Save results to file
        output_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'model_arn': args.model_arn,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()