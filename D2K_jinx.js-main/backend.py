from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from typing import List, Optional, Dict, Any
import uuid
from pathlib import Path
import logging
import tempfile
import json
import re
from fastapi.responses import FileResponse
from report_generator import generate_pdf_report
from prompts import EXTRACTION_PROMPT
import pandas as pd

load_dotenv()

# Configure the Gemini API using the newer library's client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],  # Adjust for production!
    allow_headers=["*"],
)

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Enhanced session storage that tracks documents and message history
chat_sessions = {}  # session_id -> chat object
session_documents = {}  # session_id -> list of document parts
session_history = {}  # session_id -> list of messages

logging.basicConfig(level=logging.INFO)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        chat_session = chat_sessions[session_id]
        user_message_content = request.messages[-1].content if request.messages else ""
        
        # Prepare message parts: first any documents in context, then the new message
        message_parts = []
        
        # Add any documents associated with this session
        if session_documents.get(session_id):
            message_parts.extend(session_documents[session_id])
        
        # Add the user's text message
        message_parts.append(types.Part.from_text(text=user_message_content))
        
        # Send the message with all context
        if len(message_parts) == 1:
            # Just a text message, no documents
            response = chat_session.send_message(user_message_content)
        else:
            # Message with document context
            response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({"role": "user", "content": user_message_content})
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        logging.exception("Error in /chat endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document", response_model=ChatResponse)
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    file_path = None
    try:
        session_id = session_id or str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Determine MIME type
        mime_type = "application/pdf"  # default
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        logging.info(f"Using MIME type: {mime_type} for {file.filename}")
        
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Create document part
        document_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type
        )
        
        # Initialize chat session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = client.chats.create(model='gemini-2.0-flash')
            session_history[session_id] = []
            session_documents[session_id] = []
        
        # Store document for future context
        session_documents[session_id].append(document_part)
        
        # Create the message with document and prompt
        message_parts = [document_part, types.Part.from_text(text=prompt)]
        
        # Send the message
        chat_session = chat_sessions[session_id]
        response = chat_session.send_message(message_parts)
        
        # Store in history
        session_history[session_id].append({
            "role": "user", 
            "content": f"[Uploaded document: {file.filename}] {prompt}"
        })
        session_history[session_id].append({"role": "assistant", "content": response.text})
        
        # Clean up file
        if file_path:
            file_path.unlink(missing_ok=True)
        
        return ChatResponse(response=response.text, session_id=session_id)
    
    except Exception as e:
        if file_path:
            file_path.unlink(missing_ok=True)
        logging.exception("Error in upload_document endpoint:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    if session_id not in session_history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": session_history[session_id]}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/generate_report")
async def generate_report(file: UploadFile = File(...)):
    """
    Endpoint to analyze a financial document and generate a PDF report.
    Directly uses the uploaded PDF file with Gemini's vision capabilities.
    """
    file_path = None
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Determine MIME type
        mime_type = "application/pdf"  # default
        if file.filename.lower().endswith(".csv"):
            mime_type = "text/csv"
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        # Create a temporary chat session for analysis
        analysis_session = client.chats.create(model='gemini-2.0-flash')
        # Read file data
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Create document part
        document_part = types.Part.from_bytes(
            data=file_data,
            mime_type=mime_type
        )
        
        # Use EXTRACTION_PROMPT from prompts.py instead of inline text
        extraction_prompt = EXTRACTION_PROMPT
        # Send the extraction request with document context
        message_parts = [document_part, types.Part.from_text(text=extraction_prompt)]
        extraction_response = analysis_session.send_message(message_parts)
        
        # Process the JSON response
        json_text = extraction_response.text
        try:
            # First try direct JSON parsing
            extracted_data = json.loads(json_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(1))
            else:
                # One more attempt without code block markers
                json_match = re.search(r'({[\s\S]*})', json_text)
                if json_match:
                    extracted_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("Failed to extract valid JSON from LLM response")
        
        # Calculate financial ratios
        from langchain_integration import LangChainHandler
        handler = LangChainHandler()
        calculated_ratios = handler.calculate_financial_ratios(extracted_data)
        
        # Add red flags detection
        red_flags_detection = handler.detect_financial_red_flags(extracted_data, calculated_ratios)
        
        # Generate key findings, sentiment analysis, and business model recommendations
        logging.info("Generating key findings, sentiment analysis, and business model recommendations...")
        key_findings_json = handler.generate_key_findings(extracted_data, calculated_ratios)
        
        # Generate business overview
        logging.info("Generating business overview...")
        business_overview = handler.generate_business_overview(extracted_data)
        
        # Log what we received from the handler
        logging.info(f"Business overview: {business_overview[:100]}...")
        logging.info(f"Key findings: {key_findings_json.get('key_findings', '')[:100]}...")
        logging.info(f"Sentiment analysis: {key_findings_json.get('sentiment_analysis', '')[:100]}...")
        logging.info(f"Business model: {key_findings_json.get('business_model', '')[:100]}...")
                
        # Merge the separately detected red flags with the ones returned from the findings prompt,
        # if necessary – here we give priority to key_findings' red_flags.
        report_data = {
            "business_overview": business_overview,
            "key_findings": key_findings_json.get("key_findings", ""),
            "sentiment_analysis": key_findings_json.get("sentiment_analysis", "No sentiment analysis available."),
            "business_model": key_findings_json.get("business_model", "No business model suggestions available."),
            "extracted_data": extracted_data,
            "calculated_ratios": calculated_ratios,
            "red_flags": key_findings_json.get("red_flags", red_flags_detection.get("red_flags", []))
        }

        logging.info("Preparing to generate PDF report with all data...")
        
        # Generate PDF report in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            output_path = tmp.name
        
        generate_pdf_report(report_data, output_path)
        logging.info(f"PDF report generated successfully at {output_path}")
        
        # Clean up the temporary file
        if file_path:
            file_path.unlink(missing_ok=True)
        
        return FileResponse(path=output_path, filename="financial_report.pdf", media_type="application/pdf")
    except Exception as e:
        # Clean up on error
        if file_path:
            file_path.unlink(missing_ok=True)
        logging.exception(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_anomalies")
async def detect_anomalies(
    file: UploadFile = File(...),
    sensitivity: float = Form(1.0)
):
    """
    Detect anomalies in financial data using statistical methods.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Save content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        with open(temp_file, "wb") as f:
            f.write(content)

        try:
            # Read financial data
            financial_data = pd.read_csv(temp_file)

            # Basic statistical anomaly detection
            anomalies = []
            for column in financial_data.select_dtypes(include=['float64', 'int64']).columns:
                mean = financial_data[column].mean()
                std = financial_data[column].std()
                threshold = sensitivity * 2 * std

                # Find rows where values are beyond threshold
                outliers = financial_data[abs(financial_data[column] - mean) > threshold]

                for idx, row in outliers.iterrows():
                    anomalies.append({
                        "Year": row.get("Year", idx),
                        "Column": column,
                        "Value": row[column],
                        "Mean": mean,
                        "Deviation": abs(row[column] - mean),
                        "Confidence": min(abs(row[column] - mean) / (3 * std), 1.0)
                    })

            return {
                "num_anomalies": len(anomalies),
                "anomalies": anomalies
            }

        except Exception as e:
            logging.exception("Error during anomaly detection processing:")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # Clean up temp file
            os.unlink(temp_file)

    except Exception as e:
        logging.exception("Error in detect_anomalies endpoint:")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

    print("API KEY:", os.getenv("GOOGLE_API_KEY"))