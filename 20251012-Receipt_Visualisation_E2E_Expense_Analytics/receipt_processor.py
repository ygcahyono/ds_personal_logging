import anthropic
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()

class ReceiptProcessor:
    def __init__(self):
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Initialize Google Sheets
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID')
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
        self.sheets_service = build('sheets', 'v4', credentials=credentials)
        
        # Setup directories
        self.receipts_dir = Path('receipts')
        self.processed_dir = self.receipts_dir / 'processed'
        self.receipts_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize sheet with headers if empty
        self.initialize_sheet()
    
    def initialize_sheet(self):
        """Create headers in Google Sheet if not exists"""
        try:
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.sheet_id,
                range='A1:K1'
            ).execute()
            
            # If empty, add headers
            if 'values' not in result:
                headers = [[
                    'Date Processed',
                    'Receipt Date',
                    'Store',
                    'Item Name',
                    'Quantity',
                    'Unit Price',
                    'Discount',
                    'Final Price',
                    'Subtotal',
                    'Total',
                    'Filename'
                ]]
                
                self.sheets_service.spreadsheets().values().update(
                    spreadsheetId=self.sheet_id,
                    range='A1:K1',
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()
                
                print("✓ Sheet initialized with headers")
        except HttpError as e:
            print(f"Error initializing sheet: {e}")
    
    def extract_receipt_data(self, image_path):
        """Use Claude to extract structured data from receipt image"""
        print(f"Processing: {image_path}")
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
        
        # Determine image type
        image_ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(image_ext, 'image/jpeg')
        
        # Prompt for Claude
        prompt = """Extract all information from this receipt and return it as a JSON object.

Required format:
{
  "store_name": "Store name",
  "receipt_date": "DD/MM/YYYY",
  "items": [
    {
      "name": "Item name",
      "quantity": 1,
      "unit_price": 0.00,
      "discount": 0.00,
      "final_price": 0.00
    }
  ],
  "subtotal": 0.00,
  "total_savings": 0.00,
  "total": 0.00,
  "payment_method": "Card type",
  "currency": "GBP"
}

Rules:
- Extract ALL items from the receipt
- Include meal deal discounts as negative values in "discount" field
- final_price = unit_price - discount
- Use 0.00 if a value is not found
- Date format must be DD/MM/YYYY
- Return ONLY valid JSON, no markdown or extra text"""

        try:
            # Call Claude API
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Extract JSON from response
            response_text = message.content[0].text
            
            # Try to parse JSON (handle potential markdown wrapping)
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            print("✓ Receipt data extracted successfully")
            return data
            
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            print(f"Response: {response_text}")
            return None
        except Exception as e:
            print(f"✗ Error extracting receipt data: {e}")
            return None
    
    def upload_to_sheets(self, receipt_data, filename):
        """Upload extracted data to Google Sheets"""
        if not receipt_data:
            print("✗ No data to upload")
            return False
        
        try:
            date_processed = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rows = []
            
            # Create a row for each item
            for item in receipt_data.get('items', []):
                row = [
                    date_processed,
                    receipt_data.get('receipt_date', ''),
                    receipt_data.get('store_name', ''),
                    item.get('name', ''),
                    item.get('quantity', 1),
                    item.get('unit_price', 0.00),
                    item.get('discount', 0.00),
                    item.get('final_price', 0.00),
                    receipt_data.get('subtotal', 0.00),
                    receipt_data.get('total', 0.00),
                    filename
                ]
                rows.append(row)
            
            # Append to sheet
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=self.sheet_id,
                range='A:K',
                valueInputOption='USER_ENTERED',
                body={'values': rows}
            ).execute()
            
            print(f"✓ Uploaded {len(rows)} rows to Google Sheets")
            return True
            
        except HttpError as e:
            print(f"✗ Error uploading to sheets: {e}")
            return False
    
    def process_receipt(self, image_path):
        """Process a single receipt: extract and upload"""
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Extract data
        receipt_data = self.extract_receipt_data(image_path)
        
        if receipt_data:
            # Upload to sheets
            success = self.upload_to_sheets(receipt_data, Path(image_path).name)
            
            if success:
                # Move to processed folder
                processed_path = self.processed_dir / Path(image_path).name
                Path(image_path).rename(processed_path)
                print(f"✓ Moved to: {processed_path}")
                print(f"✓ Complete!\n")
                return True
        
        print(f"✗ Failed to process receipt\n")
        return False
    
    def process_all_receipts(self):
        """Process all receipts in the receipts folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        receipt_files = [
            f for f in self.receipts_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not receipt_files:
            print("No receipts found in 'receipts' folder")
            return
        
        print(f"Found {len(receipt_files)} receipt(s) to process\n")
        
        successful = 0
        for receipt_file in receipt_files:
            if self.process_receipt(receipt_file):
                successful += 1
        
        print(f"\n{'='*60}")
        print(f"Summary: {successful}/{len(receipt_files)} receipts processed successfully")
        print(f"{'='*60}\n")


class ReceiptWatcher(FileSystemEventHandler):
    """Watch for new receipt images and auto-process"""
    def __init__(self, processor):
        self.processor = processor
        self.processing = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image
        if file_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
            return
        
        # Avoid processing the same file twice
        if str(file_path) in self.processing:
            return
        
        self.processing.add(str(file_path))
        
        # Wait a bit to ensure file is fully written
        time.sleep(1)
        
        # Process the receipt
        self.processor.process_receipt(file_path)
        
        self.processing.discard(str(file_path))


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("Receipt Processor - Claude API + Google Sheets")
    print("="*60 + "\n")
    
    # Check environment variables
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("✗ Error: ANTHROPIC_API_KEY not found in .env file")
        return
    
    if not os.getenv('GOOGLE_SHEET_ID'):
        print("✗ Error: GOOGLE_SHEET_ID not found in .env file")
        return
    
    processor = ReceiptProcessor()
    
    print("Choose mode:")
    print("1. Process all receipts in 'receipts' folder (one-time)")
    print("2. Watch 'receipts' folder and auto-process new receipts")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        processor.process_all_receipts()
    
    elif choice == '2':
        print("\n👁️  Watching 'receipts' folder for new files...")
        print("Drop receipt images here and they'll be auto-processed")
        print("Press Ctrl+C to stop\n")
        
        event_handler = ReceiptWatcher(processor)
        observer = Observer()
        observer.schedule(event_handler, str(processor.receipts_dir), recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n\n✓ Stopped watching")
        
        observer.join()
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()