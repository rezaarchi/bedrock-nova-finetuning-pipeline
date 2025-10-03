import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class SupportTicketGenerator:
    """Generate synthetic customer support ticket data for ML training"""
    
    def __init__(self, num_records=100000):
        self.num_records = num_records
        self.start_date = datetime.now() - timedelta(days=180)

        # Product features/areas
        self.product_areas = [
            'authentication',
            'billing',
            'data-export',
            'mobile-app',
            'web-dashboard',
            'api-integration',
            'notifications',
            'user-management',
            'reporting',
            'search-functionality',
            'file-upload',
            'payment-processing'
        ]

        # Ticket templates based on common support issues
        self.ticket_templates = {
            'Account Access': {
                'titles': [
                    'Unable to log in - password reset not working',
                    'Account locked after multiple failed login attempts',
                    'Two-factor authentication not receiving codes',
                    'Email verification link expired',
                    'SSO integration failing for enterprise account'
                ],
                'descriptions': [
                    'Customer reports they cannot access their account. Password reset email received but link shows "expired" error. Last successful login was {days} days ago.',
                    'User account automatically locked after {attempts} failed login attempts. Customer requesting immediate unlock. Business impact: cannot access critical data.',
                    '2FA codes not arriving via SMS. Customer tried {attempts} times. Phone number verified as correct in profile settings.',
                    'New user signup completed but verification email link returns 404 error. Customer attempted signup {hours} hours ago.',
                    'Enterprise SSO integration throwing "invalid_issuer" error. Affects {users} users in organization. Integration worked fine until {days} days ago.'
                ]
            },
            'Billing Issue': {
                'titles': [
                    'Incorrect charge on credit card - double billing',
                    'Subscription not cancelled despite confirmation',
                    'Upgrade to premium plan not reflected in account',
                    'Invoice missing - need copy for expense report',
                    'Annual plan charged monthly rate instead'
                ],
                'descriptions': [
                    'Customer charged ${amount} twice on {date}. First charge at {time1}, second at {time2}. Only one charge should have occurred. Requesting immediate refund.',
                    'Customer cancelled subscription on {date} but was charged ${amount} on renewal date. Confirmation email received but billing continued.',
                    'Upgraded from Basic to Premium {days} days ago. Payment processed successfully (${amount}) but account still shows Basic tier features.',
                    'Customer needs invoice #INV-{invoice_num} for expense reporting. Invoice not appearing in account dashboard under billing history.',
                    'Customer on annual plan (should be ${annual_amount}/year) but being charged ${monthly_amount} monthly. Contract states annual billing.'
                ]
            },
            'Feature Request': {
                'titles': [
                    'Request: Add bulk export functionality',
                    'Feature request: Dark mode for web dashboard',
                    'Suggestion: Email notification customization',
                    'Enhancement: API rate limit increase option',
                    'Request: Mobile app offline mode'
                ],
                'descriptions': [
                    'Customer managing {count} records would like ability to export all data at once. Current export limit is {limit} records. Business use case: quarterly reporting.',
                    'User requesting dark mode option for dashboard. Reports eye strain during extended sessions. Would increase product usability for {hours}+ hour daily usage.',
                    'Request to customize which events trigger email notifications. Currently all {count} notification types enabled/disabled together. Needs granular control.',
                    'Developer hitting API rate limit of {limit} requests/minute. Willing to upgrade to higher tier for increased limit. Current integration requires {needed} requests/minute.',
                    'Field teams need offline access to mobile app. Current implementation requires constant internet connection. Critical for {use_case} use case.'
                ]
            },
            'Technical Bug': {
                'titles': [
                    'Search functionality returning no results',
                    'File upload fails for files larger than 10MB',
                    'Dashboard charts not rendering in Chrome browser',
                    'Mobile app crashing on Android 14 devices',
                    'API returning 500 errors intermittently'
                ],
                'descriptions': [
                    'Search feature returns zero results for queries that should match existing records. Tested with {count} different queries. Works in Safari but not Chrome/Firefox.',
                    'Upload fails for files over {size}MB with error "upload_failed". Files under {size}MB work fine. File types: PDF, DOCX. Browser: Chrome {version}.',
                    'Dashboard data visualization charts not displaying. Blank white space where charts should appear. Issue started {days} days ago. Works in Safari/Firefox, broken in Chrome.',
                    'Mobile app crashes on launch for Android 14 users. Crash occurs within {seconds} seconds of opening app. Android 13 and below working normally. {users} users affected.',
                    'API endpoint /api/v2/data returning HTTP 500 errors approximately {percent}% of the time. No pattern to when errors occur. Other endpoints working normally.'
                ]
            },
            'Data Issue': {
                'titles': [
                    'Imported data not appearing in dashboard',
                    'Export file contains incorrect/corrupted data',
                    'Data synchronization delay between systems',
                    'Records showing duplicate entries after migration',
                    'Historical data missing from reports'
                ],
                'descriptions': [
                    'Customer imported CSV file with {count} records {hours} hours ago. Import process showed "success" but records not visible in dashboard or search results.',
                    'Exported data file contains {percent}% corrupted records. Values showing as "NULL" or random characters. Same records display correctly in web interface.',
                    'Changes made in System A taking {minutes} minutes to appear in System B. Expected sync time is under 5 seconds. Started occurring {days} days ago.',
                    'After data migration {date}, showing {count} duplicate records. Each record appears {duplicates} times. Duplicates have identical data except timestamps.',
                    'Reports for date range {start_date} to {end_date} showing incomplete data. Records from {missing_date} completely missing. Database shows records exist.'
                ]
            },
            'Performance Issue': {
                'titles': [
                    'Dashboard loading extremely slowly',
                    'Search queries timing out after 30 seconds',
                    'Report generation taking over 10 minutes',
                    'Mobile app very laggy and unresponsive',
                    'API response times increased significantly'
                ],
                'descriptions': [
                    'Dashboard taking {seconds} seconds to load. Previously loaded in under 5 seconds. Issue started {days} days ago. Tested on multiple browsers/devices.',
                    'Search functionality timing out. Queries that previously returned results in 2-3 seconds now taking over {seconds} seconds or failing entirely.',
                    'Generating monthly report taking {minutes} minutes. Report contains {count} records. Same report last month took under 2 minutes. No data volume change.',
                    'Mobile app (iOS version {version}) extremely laggy. Actions like button taps taking {seconds} seconds to register. App version {old_version} performed normally.',
                    'API endpoint response times increased from average {old_time}ms to {new_time}ms. No changes to request volume or query complexity. Started {days} days ago.'
                ]
            },
            'General Inquiry': {
                'titles': [
                    'How to configure email notifications?',
                    'Question about data retention policy',
                    'Requesting documentation for API integration',
                    'How to add team members to account?',
                    'What features included in Premium plan?'
                ],
                'descriptions': [
                    'Customer asking how to configure notification settings. Wants to enable email alerts for {event_type} events but disable for others. Looking for step-by-step instructions.',
                    'Customer inquiry about data retention policy. Specifically asking how long data is stored and if there is archival option for older than {days} days.',
                    'Developer requesting comprehensive API documentation. Specifically needs information about authentication, rate limits, and available endpoints.',
                    'Account admin asking how to invite team members. Current team size: {count}. Wants to add {new_count} additional users. Needs to understand permission levels.',
                    'Customer on Basic plan considering upgrade. Requesting detailed comparison of Premium vs Basic features. Specific interest in {feature1} and {feature2}.'
                ]
            }
        }

        # Support teams
        self.support_teams = {
            'Account Access': ['Identity & Access Team', 'Customer Success', 'Security Team'],
            'Billing Issue': ['Billing Support', 'Finance Team', 'Account Management'],
            'Feature Request': ['Product Team', 'Engineering', 'Customer Success'],
            'Technical Bug': ['Engineering Team', 'QA Team', 'Technical Support'],
            'Data Issue': ['Data Operations', 'Engineering Team', 'Technical Support'],
            'Performance Issue': ['Infrastructure Team', 'Engineering Team', 'SRE Team'],
            'General Inquiry': ['Customer Support', 'Customer Success', 'Documentation Team']
        }

    def generate_ticket(self, index):
        # Time-based patterns
        ticket_time = self.start_date + timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        # More tickets during business hours
        if 9 <= ticket_time.hour <= 17 and ticket_time.weekday() < 5:
            category = np.random.choice(
                list(self.ticket_templates.keys()),
                p=[0.15, 0.15, 0.1, 0.2, 0.15, 0.15, 0.1]
            )
        else:
            category = np.random.choice(list(self.ticket_templates.keys()))

        template = self.ticket_templates[category]
        title_template = random.choice(template['titles'])
        desc_template = random.choice(template['descriptions'])

        # Fill in variables
        variables = {
            'days': random.randint(1, 30),
            'hours': random.randint(1, 48),
            'minutes': random.randint(5, 120),
            'seconds': random.randint(10, 120),
            'attempts': random.randint(3, 10),
            'users': random.randint(10, 500),
            'amount': random.randint(10, 500),
            'count': random.randint(100, 10000),
            'limit': random.randint(100, 1000),
            'size': random.randint(5, 50),
            'percent': random.randint(10, 90),
            'needed': random.randint(1000, 5000),
            'duplicates': random.randint(2, 5),
            'invoice_num': random.randint(10000, 99999),
            'annual_amount': random.choice([499, 999, 1999, 2999]),
            'monthly_amount': random.choice([49, 99, 199, 299]),
            'version': f'{random.randint(100, 120)}.0',
            'old_version': f'{random.randint(90, 99)}.0',
            'old_time': random.randint(100, 300),
            'new_time': random.randint(1000, 5000),
            'date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'time1': f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}',
            'time2': f'{random.randint(0, 23):02d}:{random.randint(0, 59):02d}',
            'start_date': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
            'end_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'missing_date': (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
            'use_case': random.choice(['field operations', 'remote work', 'travel scenarios']),
            'event_type': random.choice(['billing', 'security', 'system', 'user activity']),
            'feature1': random.choice(['advanced analytics', 'API access', 'custom branding']),
            'feature2': random.choice(['priority support', 'increased storage', 'team collaboration']),
            'new_count': random.randint(5, 20)
        }

        title = title_template.format(**variables)
        description = desc_template.format(**variables)

        # Severity/Priority
        severity_options = ['Critical', 'High', 'Medium', 'Low']
        severity = np.random.choice(severity_options, p=[0.1, 0.25, 0.40, 0.25])

        # Resolution time based on severity
        resolution_hours = {
            'Critical': random.uniform(0.5, 4),
            'High': random.uniform(2, 12),
            'Medium': random.uniform(4, 48),
            'Low': random.uniform(12, 168)
        }

        resolved_time = ticket_time + timedelta(hours=resolution_hours[severity])

        # Generate resolution
        resolutions = {
            'Account Access': [
                'Password reset link regenerated and sent. Customer able to access account successfully.',
                'Account unlocked. Added IP address to allowlist. Customer confirmed access restored.',
                'Identified SMS gateway delay. Switched to email 2FA. Customer logged in successfully.',
                'Email verification system had caching issue. Cleared cache, new link sent. Signup completed.',
                'SSO configuration updated. Issuer URL corrected in settings. All {users} users can now authenticate.'
            ],
            'Billing Issue': [
                'Duplicate charge refunded. ${amount} credited back to card within 3-5 business days.',
                'Subscription cancellation processed retroactively. ${amount} refunded. Confirmation email sent.',
                'Account tier manually upgraded to Premium. Features now accessible. System sync issue resolved.',
                'Invoice #INV-{invoice_num} regenerated and emailed. Also available in dashboard billing section.',
                'Billing plan corrected to annual. Future charges will be ${annual_amount}/year. Partial refund issued.'
            ],
            'Feature Request': [
                'Feature request logged as FR-{ticket_num}. Added to product roadmap for Q{quarter} evaluation.',
                'Dark mode feature already in development. Expected release in {weeks} weeks. Added customer to beta program.',
                'Notification customization feature prioritized. Engineering team assigned. Estimated delivery: {weeks} weeks.',
                'Enterprise tier created with {limit} requests/minute limit. Customer upgraded. Rate limit increased.',
                'Offline mode requirement documented. Added to mobile app roadmap. Will notify when available.'
            ],
            'Technical Bug': [
                'Search index rebuilt. Missing records re-indexed. Search functionality now returning correct results.',
                'File upload limit increased to {size}MB. Server configuration updated. Uploads now working.',
                'Chrome rendering bug fixed. Deployed patch to production. Charts now displaying correctly.',
                'Android 14 compatibility issue identified. App update v{version} released. Crash resolved.',
                'API 500 error traced to database connection pool exhaustion. Pool size increased. Errors resolved.'
            ],
            'Data Issue': [
                'Import job re-run manually. All {count} records now visible. Background job queue was stuck.',
                'Export bug fixed. Data integrity check added to export process. Re-exported file sent to customer.',
                'Sync delay caused by network latency. Optimized sync protocol. Latency reduced to under 5 seconds.',
                'Duplicate records identified and merged. De-duplication script run. {count} unique records remain.',
                'Missing data restored from backup. Records for {missing_date} now showing in reports.'
            ],
            'Performance Issue': [
                'Database query optimized. Added caching layer. Dashboard load time reduced to under 3 seconds.',
                'Search timeout increased to {seconds} seconds. Query optimizer improved. Searches now completing.',
                'Report generation parallelized. Processing time reduced from {old_minutes} to {new_minutes} minutes.',
                'Mobile app performance issue fixed in v{version}. Memory leak resolved. Update available in app store.',
                'API infrastructure scaled horizontally. Response times back to {old_time}ms average. Monitoring added.'
            ],
            'General Inquiry': [
                'Step-by-step notification configuration guide sent via email. Customer confirmed settings updated successfully.',
                'Data retention policy documentation provided. Data stored for {days} days with archival options explained.',
                'Complete API documentation sent. Integration guide included. Developer confirmed sufficient for integration.',
                'Team member invitation guide provided. Customer successfully added {new_count} users with appropriate permissions.',
                'Feature comparison chart sent. Explained Premium benefits. Customer upgraded to Premium plan.'
            ]
        }

        resolution_template = random.choice(resolutions[category])
        resolution = resolution_template.format(
            **variables,
            ticket_num=random.randint(1000, 9999),
            quarter=random.randint(2, 4),
            weeks=random.randint(4, 12),
            old_minutes=random.randint(10, 20),
            new_minutes=random.randint(2, 5)
        )

        ticket_id = f'SUP-{random.randint(100000, 999999)}'

        return {
            'TICKET_ID': ticket_id,
            'TICKET_TITLE': title,
            'TICKET_DESCRIPTION': description,
            'TICKET_TYPE': 'Support Request',
            'PRIORITY': f'P{random.randint(1, 4)}',
            'SEVERITY': severity,
            'STATUS': 'Resolved',
            'RESOLUTION_DESCRIPTION': resolution,
            'CATEGORY': category,
            'SUBCATEGORY': random.choice(self.product_areas),
            'OPEN_DATETIME': ticket_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'CLOSE_DATETIME': resolved_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'ASSIGNED_TEAM': random.choice(self.support_teams[category]),
            'ASSIGNEE_NAME': f'Agent{random.randint(1, 50)}',
            'CUSTOMER_TIER': random.choice(['Free', 'Basic', 'Premium', 'Enterprise']),
            'CHANNEL': random.choice(['Email', 'Chat', 'Phone', 'Web Form', 'API']),
            'SATISFACTION_SCORE': random.randint(3, 5) if resolved_time else None
        }

    def generate_dataset(self):
        tickets = []

        for i in range(self.num_records):
            ticket = self.generate_ticket(i)
            tickets.append(ticket)

            if i % 10000 == 0:
                print(f"Generated {i} tickets...")

        df = pd.DataFrame(tickets)

        # Add correlation patterns
        monday_mask = pd.to_datetime(df['OPEN_DATETIME']).dt.dayofweek == 0
        morning_mask = pd.to_datetime(df['OPEN_DATETIME']).dt.hour.between(8, 10)
        df.loc[monday_mask & morning_mask, 'CATEGORY'] = np.random.choice(
            ['Account Access', 'Technical Bug'], 
            size=sum(monday_mask & morning_mask)
        )

        return df

if __name__ == '__main__':
    import sys
    
    num_records = 100000
    
    if len(sys.argv) > 1:
        try:
            num_records = int(sys.argv[1])
            print(f"Generating {num_records} records")
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python generate_support_data.py [num_records]")
            sys.exit(1)
    
    generator = SupportTicketGenerator(num_records=num_records)
    df = generator.generate_dataset()

    df.to_csv('support_tickets_training_data.csv', index=False)

    print("\nDataset Generated Successfully!")
    print(f"Total Records: {len(df)}")
    print("\nCategory Distribution:")
    print(df['CATEGORY'].value_counts())
    print("\nSeverity Distribution:")
    print(df['SEVERITY'].value_counts())
    print("\nCustomer Tier Distribution:")
    print(df['CUSTOMER_TIER'].value_counts())

    sample_size = min(100, len(df))
    df.head(sample_size).to_excel('support_tickets_sample.xlsx', index=False)
    print(f"\nSample data ({sample_size} records) saved to support_tickets_sample.xlsx")