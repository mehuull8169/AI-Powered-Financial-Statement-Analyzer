import os
import requests
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_report_generation():
    """Test the report generation functionality with a sample PDF."""
    
    # Path to a sample financial document (PDF)
    test_file_path = Path("test_files/sample_financial_statement.pdf")
    
    if not test_file_path.exists():
        logging.error(f"Test file not found: {test_file_path}")
        return
    
    try:
        logging.info("Testing report generation with sample file")
        
        # Create a multipart form request with the test PDF
        files = {'file': (test_file_path.name, open(test_file_path, 'rb'), 'application/pdf')}
        
        logging.info("Sending request to /generate_report endpoint")
        response = requests.post(
            'http://127.0.0.1:8000/generate_report',
            files=files
        )
        
        if response.status_code == 200:
            logging.info("Report generated successfully!")
            
            # Save the PDF report
            output_path = Path("test_output/test_report.pdf") 
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"Report saved to {output_path}")
            return True
        else:
            logging.error(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.exception(f"Test failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    # Create test directories if they don't exist
    Path("test_files").mkdir(exist_ok=True)
    Path("test_output").mkdir(exist_ok=True)
    
    # Print a warning if the test file doesn't exist
    if not Path("test_files/sample_financial_statement.pdf").exists():
        print("Please add a sample financial statement PDF to the 'test_files' directory")
        print("named 'sample_financial_statement.pdf' before running this test.")
    
    test_result = test_report_generation()
    print(f"Test {'PASSED' if test_result else 'FAILED'}")
