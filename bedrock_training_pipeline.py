"""
Amazon Bedrock Nova Pro Fine-Tuning Pipeline for Support Ticket Classification
Complete script to create training data, format it, create AWS resources, and fine-tune model
"""

import boto3
import json
import pandas as pd
import time
from datetime import datetime
import os
from typing import Dict, List

class BedrockSupportPipeline:
    def __init__(self, region='us-east-1', config_file='bedrock_pipeline_config.json'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.bedrock_client = boto3.client('bedrock', region_name=region)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.iam_client = boto3.client('iam')
        
        # Configuration file for state persistence
        self.config_file = config_file
        self.config = self._load_config()
        
        # Use existing or create new identifiers
        timestamp = int(time.time())
        self.bucket_name = self.config.get('bucket_name', f'support-bedrock-training-{timestamp}')
        self.model_name = self.config.get('model_name', f'support-classifier-{timestamp}')
        self.role_arn = self.config.get('role_arn')
        self.role_name = self.config.get('role_name')
        
        # Correct base model ID for Nova Pro
        self.base_model_id = self.config.get('base_model_id', 'amazon.nova-pro-v1:0')
        
        print(f"Initialized pipeline in region: {region}")
        print(f"S3 Bucket: {self.bucket_name}")
        print(f"Model Name: {self.model_name}")
        print(f"Base Model: {self.base_model_id}")
        if self.role_arn:
            print(f"Using existing IAM Role: {self.role_arn}")
    
    def _load_config(self):
        """Load existing configuration if available"""
        if os.path.exists(self.config_file):
            print(f"✓ Loading existing configuration from {self.config_file}")
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_config(self):
        """Save configuration for reuse"""
        config_data = {
            'bucket_name': self.bucket_name,
            'model_name': self.model_name,
            'role_arn': self.role_arn,
            'role_name': self.role_name,
            'region': self.region,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"✓ Configuration saved to {self.config_file}")
    
    def list_available_models(self):
        """List available foundation models for fine-tuning"""
        print("\n=== Checking Available Models ===")
        try:
            response = self.bedrock_client.list_foundation_models(
                byCustomizationType='FINE_TUNING'
            )
            
            print("Available models for fine-tuning:")
            nova_models = []
            for model in response.get('modelSummaries', []):
                model_id = model['modelId']
                model_name = model.get('modelName', 'Unknown')
                print(f"  - {model_id} ({model_name})")
                if 'nova' in model_id.lower():
                    nova_models.append(model_id)
            
            if nova_models:
                print(f"\nNova models found: {nova_models}")
                return nova_models[0]
            else:
                print("\nNo Nova models found. Available models listed above.")
                return None
                
        except Exception as e:
            print(f"Error listing models: {e}")
            return None

    def create_s3_bucket(self):
        """Create S3 bucket for training data"""
        print(f"\n=== Creating S3 Bucket: {self.bucket_name} ===")
        
        try:
            # Check if bucket already exists
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                print(f"✓ S3 bucket already exists: {self.bucket_name}")
                self._save_config()
                return self.bucket_name
            except:
                pass
            
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            print(f"✓ S3 bucket created successfully")
            self._save_config()
            return self.bucket_name
        except Exception as e:
            print(f"Error creating bucket: {e}")
            raise

    def create_iam_role(self):
        """Create IAM role for Bedrock training"""
        print("\n=== Creating IAM Role for Bedrock ===")
        
        if self.role_arn:
            try:
                self.iam_client.get_role(RoleName=self.role_name)
                print(f"✓ Using existing IAM role: {self.role_arn}")
                return self.role_arn
            except:
                pass
        
        role_name = f'BedrockSupportTrainingRole-{int(time.time())}'
        self.role_name = role_name
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        
        permission_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{self.bucket_name}",
                    f"arn:aws:s3:::{self.bucket_name}/*"
                ]
            }]
        }
        
        try:
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for Bedrock Support Ticket Classification training'
            )
            
            self.role_arn = response['Role']['Arn']
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='BedrockS3Access',
                PolicyDocument=json.dumps(permission_policy)
            )
            
            print(f"✓ IAM role created: {self.role_arn}")
            print("  Waiting for IAM role to propagate...")
            time.sleep(10)
            
            self._save_config()
            return self.role_arn
        except Exception as e:
            print(f"Error creating IAM role: {e}")
            raise

    def prepare_training_data(self, df: pd.DataFrame, train_ratio=0.8, max_samples=10000):
        """Convert CSV data to JSONL format for Bedrock"""
        print("\n=== Preparing Training Data ===")
        
        # Nova Pro has a 10,000 sample limit for training
        # Since we create 2 examples per ticket (classification + resolution),
        # we need to limit input tickets accordingly
        max_tickets = max_samples // 2  # Reserve room for both example types
        
        if len(df) > max_tickets:
            print(f"⚠️  Dataset has {len(df)} tickets, but Nova Pro limit is {max_samples} training samples")
            print(f"   Sampling {max_tickets} tickets to stay within limits...")
            df = df.sample(n=max_tickets, random_state=42)
        
        total_records = len(df)
        train_size = int(total_records * train_ratio)
        
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        print(f"Total tickets: {total_records}")
        print(f"Training tickets: {len(train_df)} ({train_ratio*100:.0f}%)")
        print(f"Validation tickets: {len(val_df)} ({(1-train_ratio)*100:.0f}%)")
        
        training_data = []
        validation_data = []
        
        print("Converting to Nova Pro format...")
        
        for idx, row in train_df.iterrows():
            if pd.isna(row['CATEGORY']) or pd.isna(row['SEVERITY']):
                continue
                
            # Classification task
            classification_example = {
                "system": [{
                    "text": "You are a support ticket classification assistant. Analyze customer support tickets and provide accurate categorization, severity assessment, and routing recommendations."
                }],
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "text": f"Classify this support ticket:\n\nTitle: {row['TICKET_TITLE']}\n\nDescription: {row['TICKET_DESCRIPTION']}\n\nProvide the category, severity, and recommended team."
                        }]
                    },
                    {
                        "role": "assistant",
                        "content": [{
                            "text": f"Category: {row['CATEGORY']}\nSeverity: {row['SEVERITY']}\nPriority: {row['PRIORITY']}\nRecommended Team: {row['ASSIGNED_TEAM']}\nCustomer Tier: {row['CUSTOMER_TIER']}"
                        }]
                    }
                ]
            }
            training_data.append(classification_example)
            
            # Resolution recommendation task
            if pd.notna(row['RESOLUTION_DESCRIPTION']) and row['RESOLUTION_DESCRIPTION']:
                resolution_example = {
                    "system": [{
                        "text": "You are a support ticket classification assistant. Analyze customer support tickets and provide accurate categorization, severity assessment, and routing recommendations."
                    }],
                    "messages": [
                        {
                            "role": "user",
                            "content": [{
                                "text": f"A support ticket was submitted:\n\nTitle: {row['TICKET_TITLE']}\nDescription: {row['TICKET_DESCRIPTION']}\nSeverity: {row['SEVERITY']}\n\nWhat steps would you recommend to resolve this?"
                            }]
                        },
                        {
                            "role": "assistant",
                            "content": [{
                                "text": f"Recommended Resolution:\n{row['RESOLUTION_DESCRIPTION']}\n\nThis ticket should be assigned to: {row['ASSIGNED_TEAM']}"
                            }]
                        }
                    ]
                }
                training_data.append(resolution_example)
        
        # Validation examples
        for idx, row in val_df.iterrows():
            if pd.isna(row['CATEGORY']) or pd.isna(row['SEVERITY']):
                continue
                
            validation_example = {
                "system": [{
                    "text": "You are a support ticket classification assistant. Analyze customer support tickets and provide accurate categorization, severity assessment, and routing recommendations."
                }],
                "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "text": f"Classify this support ticket:\n\nTitle: {row['TICKET_TITLE']}\n\nDescription: {row['TICKET_DESCRIPTION']}\n\nProvide the category, severity, and recommended team."
                        }]
                    },
                    {
                        "role": "assistant",
                        "content": [{
                            "text": f"Category: {row['CATEGORY']}\nSeverity: {row['SEVERITY']}\nPriority: {row['PRIORITY']}\nRecommended Team: {row['ASSIGNED_TEAM']}\nCustomer Tier: {row['CUSTOMER_TIER']}"
                        }]
                    }
                ]
            }
            validation_data.append(validation_example)
        
        # Check if we're within limits
        if len(training_data) > max_samples:
            print(f"⚠️  Warning: Generated {len(training_data)} training samples, trimming to {max_samples}")
            training_data = training_data[:max_samples]
        
        print(f"✓ Created {len(training_data)} training examples (limit: {max_samples})")
        print(f"✓ Created {len(validation_data)} validation examples")
        
        if len(training_data) < 8:
            raise ValueError(f"Not enough training samples ({len(training_data)}). Minimum is 8.")
        
        # Save to JSONL files
        with open('training_data.jsonl', 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        with open('validation_data.jsonl', 'w') as f:
            for item in validation_data:
                f.write(json.dumps(item) + '\n')
        
        print("\nValidating JSONL format...")
        self._validate_jsonl_format('training_data.jsonl')
        self._validate_jsonl_format('validation_data.jsonl')
        
        return 'training_data.jsonl', 'validation_data.jsonl'
    
    def _validate_jsonl_format(self, filename: str):
        """Validate JSONL file format"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines[:3]):
                data = json.loads(line)
                assert 'system' in data, f"Missing 'system' in line {i+1}"
                assert 'messages' in data, f"Missing 'messages' in line {i+1}"
                assert len(data['messages']) >= 2, f"Need 2+ messages in line {i+1}"
            
            print(f"✓ {filename} format is valid")
        except Exception as e:
            print(f"✗ Validation error in {filename}: {e}")
            raise

    def upload_to_s3(self, train_file: str, val_file: str):
        """Upload training data to S3"""
        print("\n=== Uploading Training Data to S3 ===")
        
        try:
            train_key = 'training/training_data.jsonl'
            self.s3_client.upload_file(train_file, self.bucket_name, train_key)
            train_s3_uri = f's3://{self.bucket_name}/{train_key}'
            print(f"✓ Uploaded training data: {train_s3_uri}")
            
            val_key = 'validation/validation_data.jsonl'
            self.s3_client.upload_file(val_file, self.bucket_name, val_key)
            val_s3_uri = f's3://{self.bucket_name}/{val_key}'
            print(f"✓ Uploaded validation data: {val_s3_uri}")
            
            return train_s3_uri, val_s3_uri
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise

    def create_fine_tuning_job(self, train_s3_uri: str, val_s3_uri: str, role_arn: str):
        """Create Bedrock fine-tuning job"""
        print("\n=== Creating Bedrock Fine-Tuning Job ===")
        
        output_s3_uri = f's3://{self.bucket_name}/output/'
        
        hyperparameters = {
            "epochCount": "3",
            "batchSize": "1",
            "learningRate": "0.00001",
            "learningRateWarmupSteps": "0"
        }
        
        try:
            response = self.bedrock_client.create_model_customization_job(
                jobName=self.model_name,
                customModelName=self.model_name,
                roleArn=role_arn,
                baseModelIdentifier=self.base_model_id,
                hyperParameters=hyperparameters,
                trainingDataConfig={'s3Uri': train_s3_uri},
                validationDataConfig={'validators': [{'s3Uri': val_s3_uri}]},
                outputDataConfig={'s3Uri': output_s3_uri}
            )
            
            job_arn = response['jobArn']
            print(f"✓ Fine-tuning job created: {job_arn}")
            print(f"\nHyperparameters:")
            for key, value in hyperparameters.items():
                print(f"  {key}: {value}")
            
            self.config['job_arn'] = job_arn
            self.config['job_name'] = self.model_name
            self._save_config()
            
            return job_arn
        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            raise

    def monitor_training_job(self, job_arn: str):
        """Monitor training job progress"""
        print("\n=== Monitoring Training Job ===")
        print("Training in progress. This may take several hours...")
        print("You can safely exit (Ctrl+C) and check status later with:")
        print(f"  aws bedrock get-model-customization-job --job-identifier {self.model_name}")
        print()
        
        job_name = job_arn.split('/')[-1]
        
        while True:
            try:
                response = self.bedrock_client.get_model_customization_job(
                    jobIdentifier=job_name
                )
                
                status = response['status']
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
                
                if status == 'InProgress':
                    if 'trainingMetrics' in response:
                        metrics = response['trainingMetrics']
                        print(f"  Training Loss: {metrics.get('trainingLoss', 'N/A')}")
                        print(f"  Validation Loss: {metrics.get('validationLoss', 'N/A')}")
                    print("  Waiting 60 seconds...")
                    time.sleep(60)
                
                elif status == 'Completed':
                    print("\n✓ Training completed successfully!")
                    if 'outputModelArn' in response:
                        model_arn = response['outputModelArn']
                        print(f"  Model ARN: {model_arn}")
                        return model_arn
                    break
                
                elif status in ['Failed', 'Stopped']:
                    print(f"\n✗ Training job {status.lower()}")
                    if 'failureMessage' in response:
                        print(f"  Error: {response['failureMessage']}")
                    break
                
            except KeyboardInterrupt:
                print("\n\nMonitoring interrupted. Training continues in background.")
                print("Check status later with above AWS CLI command.")
                return None
            except Exception as e:
                print(f"Error monitoring job: {e}")
                break

    def test_fine_tuned_model(self, model_arn: str):
        """Test the fine-tuned model"""
        print("\n=== Model Training Complete ===")
        print(f"Model ARN: {model_arn}")
        print("\nTo test the model, use the following code:")
        print(f"""
import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime')

test_ticket = '''Classify this support ticket:

Title: Cannot upload files larger than 5MB
Description: User trying to upload PDF files. Files under 5MB work fine but larger files fail with timeout error. Using Chrome browser version 120.

Provide the category, severity, and recommended team.'''

response = bedrock_runtime.invoke_model(
    modelId='{model_arn}',
    body=json.dumps({{
        "messages": [{{"role": "user", "content": [{{"text": test_ticket}}]}}],
        "inferenceConfig": {{"temperature": 0.5, "maxTokens": 512}}
    }})
)

result = json.loads(response['body'].read())
print(result['output']['message']['content'][0]['text'])
""")

    def cleanup_resources(self, delete_bucket=False, delete_role=False):
        """Cleanup resources"""
        print("\n=== Resource Management ===")
        print(f"S3 Bucket: {self.bucket_name}")
        print(f"IAM Role: {self.role_name}")
        print(f"Config File: {self.config_file}")
        
        if not delete_bucket:
            print(f"\nTo delete bucket: aws s3 rb s3://{self.bucket_name} --force")
        
        if not delete_role:
            print(f"To delete role:")
            print(f"  aws iam delete-role-policy --role-name {self.role_name} --policy-name BedrockS3Access")
            print(f"  aws iam delete-role --role-name {self.role_name}")

    def run_pipeline(self, csv_file: str):
        """Run the complete training pipeline"""
        print("=" * 70)
        print("Amazon Bedrock Fine-Tuning Pipeline")
        print("Use Case: Customer Support Ticket Classification & Routing")
        print("=" * 70)
        
        try:
            print("\n=== Verifying Model Availability ===")
            available_model = self.list_available_models()
            if available_model and available_model != self.base_model_id:
                print(f"\nUsing discovered model: {available_model}")
                self.base_model_id = available_model
                self.config['base_model_id'] = available_model
                self._save_config()
            
            self.create_s3_bucket()
            role_arn = self.create_iam_role()
            
            print("\n=== Loading Training Data ===")
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} support tickets from {csv_file}")
            
            train_file, val_file = self.prepare_training_data(df)
            train_s3_uri, val_s3_uri = self.upload_to_s3(train_file, val_file)
            job_arn = self.create_fine_tuning_job(train_s3_uri, val_s3_uri, role_arn)
            model_arn = self.monitor_training_job(job_arn)
            
            if model_arn:
                self.test_fine_tuned_model(model_arn)
            
            self.cleanup_resources()
            
            print("\n" + "=" * 70)
            print("Pipeline completed successfully!")
            print("=" * 70)
            
            return {
                'bucket': self.bucket_name,
                'role_arn': role_arn,
                'job_arn': job_arn,
                'model_arn': model_arn,
                'base_model_id': self.base_model_id
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            raise


def main():
    REGION = 'us-east-1'
    CSV_FILE = 'support_tickets_training_data.csv'
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: Training data file '{CSV_FILE}' not found!")
        print("Run: python generate_support_data.py")
        return
    
    pipeline = BedrockSupportPipeline(region=REGION)
    
    if os.path.exists(pipeline.config_file):
        print("\n" + "="*70)
        print("Existing configuration detected!")
        print("="*70)
        response = input("Use existing resources? (y/N): ").strip().lower()
        if response != 'y':
            os.remove(pipeline.config_file)
            pipeline = BedrockSupportPipeline(region=REGION)
    
    try:
        results = pipeline.run_pipeline(CSV_FILE)
        print("\n=== Pipeline Results ===")
        print(json.dumps(results, indent=2))
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. Progress saved. Run again to continue.")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        print("Progress saved. Fix the issue and run again.")


if __name__ == '__main__':
    main()