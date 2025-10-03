# Amazon Bedrock Fine-Tuning Demo

Experimental implementation demonstrating how to fine-tune Amazon Bedrock's Nova Pro model for support ticket classification. This is a learning resource for understanding Bedrock's fine-tuning capabilities.

**This is experimental code for educational purposes only. Not suitable for production use.**

## Overview

This project demonstrates:
- Generating synthetic training data
- Formatting data for Bedrock fine-tuning (JSONL)
- Automated AWS resource creation (S3, IAM)
- Creating and monitoring fine-tuning jobs
- Testing custom models

## Architecture

<img width="3604" height="2684" alt="image" src="https://github.com/user-attachments/assets/3ab0b85d-671a-4338-83f8-e010f60956dd" />

## Prerequisites

- AWS Account with Bedrock access enabled
- Python 3.13
- AWS CLI configured
- `boto3`, `pandas`, `numpy` installed

## Quick Start

### 1. Install Dependencies

```bash
pip install boto3 pandas numpy openpyxl
```

### 2. Generate Training Data

```bash
# Generate 5,000 tickets (maximum for Nova Pro)
python generate_support_data.py 5000
```

This creates:
- `support_tickets_training_data.csv` - Training dataset
- `support_tickets_sample.xlsx` - Sample for review

### 3. Run Training Pipeline

```bash
python bedrock_training_pipeline.py
```

The pipeline will:
- Create S3 bucket
- Create IAM role with required permissions
- Convert CSV to JSONL format
- Upload to S3
- Start Bedrock fine-tuning job
- Monitor training progress (3-6 hours)

### 4. Test the Model

```bash
# After training completes
python test_model.py --model-arn <your-model-arn>

# Interactive testing
python test_model.py --model-arn <your-model-arn> --interactive
```

## Important Limitations

### Nova Pro Constraints
- **Maximum training samples: 10,000**
- **Effective maximum: 5,000 tickets** (pipeline creates 2 examples per ticket)
- **Minimum samples: 8**
- **Batch size: Must be 1** (not configurable)

### Not Production Ready
This code lacks:
- Enterprise security controls
- High availability architecture
- Production monitoring/alerting
- Data governance frameworks
- Compliance certifications
- SLA guarantees

**For production AI solutions, contact [IBM](https://www.ibm.com) for enterprise implementations.**

## Project Structure

```
.
├── generate_support_data.py          # Synthetic data generator
├── bedrock_training_pipeline.py      # Main training pipeline
├── test_model.py                      # Model testing script
└── README.md
```

## Configuration

The pipeline uses `bedrock_pipeline_config.json` to track resources:

```json
{
  "bucket_name": "support-bedrock-training-1234567890",
  "model_name": "support-classifier-1234567890",
  "role_arn": "arn:aws:iam::...",
  "base_model_id": "amazon.nova-pro-v1:0"
}
```

This allows resuming interrupted runs without creating duplicate resources.

## Training Data Format

Bedrock Nova requires conversational JSONL:

```json
{
  "system": [{"text": "You are a support ticket classification assistant..."}],
  "messages": [
    {
      "role": "user",
      "content": [{"text": "Classify this ticket: ..."}]
    },
    {
      "role": "assistant",
      "content": [{"text": "Category: Technical Bug\nSeverity: High..."}]
    }
  ]
}
```

## Cost Estimate

- **Training**: ~$40-50 (one-time, for 5K tickets)
- **Testing**: ~$1 (100 test inferences)
- **S3 Storage**: ~$5/month

Total experimental cost: ~$50-100

## Troubleshooting

### "Number of samples X out of bounds between 8 and 10000"
- You exceeded Nova Pro's limit
- Generate maximum 5,000 tickets (not more)

### "Invalid model identifier"
- Check available models: `python check_bedrock_models.py`
- Correct identifier: `amazon.nova-pro-v1:0`

### "Insufficient permissions"
- Verify IAM role has S3 read/write access
- Check role trust policy allows `bedrock.amazonaws.com`

### Job Status "Not Started"
- Normal - validation happens first (15-30 min queue time)
- Training starts after validation completes

## AWS Resources Created

The pipeline creates:
- S3 bucket: `support-bedrock-training-{timestamp}`
- IAM role: `BedrockSupportTrainingRole-{timestamp}`
- Bedrock custom model: `support-classifier-{timestamp}`

## Cleanup

```bash
# Delete S3 bucket
aws s3 rb s3://your-bucket-name --force

# Delete IAM role
aws iam delete-role-policy --role-name YourRoleName --policy-name BedrockS3Access
aws iam delete-role --role-name YourRoleName

# Delete config
rm bedrock_pipeline_config.json
```
Custom models can be deleted from AWS Bedrock Console.

## Contributing

This is an educational demo. For improvements or issues, please open a GitHub issue.

## Disclaimer

This is experimental code for learning purposes. Do not use in production environments. The code is provided as-is without warranties. For production AI solutions, consult enterprise AI vendors.

## Author

Reza Beykzadeh

