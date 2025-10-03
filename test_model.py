#!/usr/bin/env python3
"""
Test script for fine-tuned Bedrock Nova Pro model
Tests the model with sample support tickets and displays results
"""

import boto3
import json
import argparse
from datetime import datetime
from typing import Dict, List

class ModelTester:
    def __init__(self, model_arn: str, region: str = 'us-east-1'):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.model_arn = model_arn
        
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
