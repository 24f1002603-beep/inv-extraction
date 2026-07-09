import json
import os
import re
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Direct integration with AI Pipe using your single environment variable token 
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI(title="Invoice Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: dict

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/")
@app.post("/extract")
def extract_invoice(req: InvoiceRequest):

    # 1. BULLETPROOF FALLBACK EXTRACTIONS (Python Safety Net)
    
    # --- Fallback: Email ---
    fallback_email = None
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        req.text
    )
    if email_match:
        fallback_email = email_match.group(0).strip().lower()

    # --- Fallback: Currency ---
    fallback_currency = None
    text_upper = req.text.upper()
    if any(x in text_upper for x in ["EUR", "EURO", "€"]): 
        fallback_currency = "EUR"
    elif any(x in text_upper for x in ["USD", "DOLLAR", "$"]): 
        fallback_currency = "USD"
    elif any(x in text_upper for x in ["GBP", "POUND", "£"]): 
        fallback_currency = "GBP"
    elif any(x in text_upper for x in ["INR", "RUPEE", "₹"]): 
        fallback_currency = "INR"
    elif any(x in text_upper for x in ["JPY", "YEN", "¥"]): 
        fallback_currency = "JPY"

    # --- Fallback: due_in_days ---
    fallback_due_days = None
    text_lower = req.text.lower()
    
    digit_days_match = re.search(r"(?:net|within|due\s+in|\bpay\b[^.!?]*?\bwithin\b)\s*(\d+)\s*day", text_lower)
    if digit_days_match:
        fallback_due_days = int(digit_days_match.group(1))
    elif "within a week" in text_lower or "due in a week" in text_lower or "payable within a week" in text_lower:
        fallback_due_days = 7
    elif "two weeks" in text_lower or "within two weeks" in text_lower or "due in two weeks" in text_lower:
        fallback_due_days = 14
    elif "net 30" in text_lower:
        fallback_due_days = 30
    elif "net 45" in text_lower:
        fallback_due_days = 45
    elif "net 60" in text_lower:
        fallback_due_days = 60

    # --- Fallback: invoice_date ---
    fallback_invoice_date = None
    iso_match = re.search(r"\b(\d{4})[-\/](\d{2})[-\/](\d{2})\b", req.text)
    if iso_match:
        fallback_invoice_date = f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
    
    if not fallback_invoice_date:
        dmy_match = re.search(r"\b(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})\b", req.text)
        if dmy_match:
            g1, g2, g3 = dmy_match.group(1).zfill(2), dmy_match.group(2).zfill(2), dmy_match.group(3)
            fallback_invoice_date = f"{g3}-{g2}-{g1}"

    if not fallback_invoice_date:
        months_regex = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        text_dmy = re.search(rf"\b(\d{{1,2}})\s+{months_regex}\s+(\d{{4}})\b", req.text, re.IGNORECASE)
        text_mdy = re.search(rf"\b{months_regex}\s+(\d{{1,2}}),?\s+(\d{{4}})\b", req.text, re.IGNORECASE)
        
        months_map = {
            "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
            "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
            "aug": "08", "august": "08", "sep": "09", "september": "09", "oct": "10", "october": "10",
            "nov": "11", "november": "11", "dec": "12", "december": "12"
        }
        if text_dmy:
            fallback_invoice_date = f"{text_dmy.group(3)}-{months_map[text_dmy.group(2).lower()]}-{text_dmy.group(1).zfill(2)}"
        elif text_mdy:
            fallback_invoice_date = f"{text_mdy.group(3)}-{months_map[text_mdy.group(1).lower()]}-{text_mdy.group(2).zfill(2)}"

    # --- Fallback: is_paid ---
    fallback_is_paid = None
    if any(x in text_lower for x in ["paid in full", "status: paid", "amount due: 0", "balance due: 00", "settled"]):
        fallback_is_paid = True
    elif any(x in text_lower for x in ["awaiting payment", "due by", "unpaid", "balance due", "please pay"]):
        fallback_is_paid = False

    # 2. CALL THE LLM WITH JSON SCHEMA ENFORCEMENT
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            max_tokens=1500, 
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "InvoiceExtractionSchema",
                    "strict": True,
                    "schema": req.schema
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction engine. You must strictly match the types "
                        "requested in the schema. Convert text descriptions of time periods or values "
                        "into clean integers (e.g., 'within a week' -> 7, 'in two weeks' -> 14, "
                        "Net 30' -> 30, 'twelve thousand' -> 12000). Always normalize invoice_date to YYYY-MM-DD. "
                        "Infer is_paid accurately as a boolean."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the matching fields from this invoice document text:\n\n{req.text}"
                }
            ],
        )

        text_res = response.choices[0].message.content
        data = json.loads(text_res)

    except Exception as e:
        print(f"Extraction processing failed: {e}")
        data = {}

    # 3. NORMALIZE & FORCE CORRECTIONS
    if not data.get("contact_email") and fallback_email:
        data["contact_email"] = fallback_email

    if not data.get("currency") and fallback_currency:
        data["currency"] = fallback_currency

    if data.get("due_in_days") is None and fallback_due_days is not None:
        data["due_in_days"] = fallback_due_days

    if (not data.get("invoice_date") or data.get("invoice_date") == "null") and fallback_invoice_date:
        data["invoice_date"] = fallback_invoice_date

    # --- The Fix for is_paid ---
    if data.get("is_paid") is None:
        if fallback_is_paid is not None:
            data["is_paid"] = fallback_is_paid
        else:
            data["is_paid"] = False # Safe default fallback instead of returning null

    # Final type casting safety checks
    if isinstance(data.get("vendor"), str):
        data["vendor"] = re.sub(r"[.,:;!?]+$", "", data["vendor"].strip())

    if isinstance(data.get("contact_email"), str):
        data["contact_email"] = data["contact_email"].strip().lower()

    if isinstance(data.get("currency"), str):
        data["currency"] = data["currency"].strip().upper()

    if isinstance(data.get("priority"), str):
        data["priority"] = data["priority"].strip().lower()

    for int_field in ["total_amount", "due_in_days"]:
        if data.get(int_field) is not None:
            try:
                data[int_field] = int(float(data[int_field]))
            except Exception:
                pass

    if isinstance(data.get("line_items"), list):
        for item in data["line_items"]:
            if not isinstance(item, dict):
                continue
            for nested_int in ["quantity", "unit_price"]:
                if item.get(nested_int) is not None:
                    try:
                        item[nested_int] = int(float(item[nested_int]))
                    except Exception:
                        pass
        data["item_count"] = len(data["line_items"])
    else:
        data["item_count"] = 0

    # 4. RETURN EXACTLY SCHEMA KEYS
    properties = req.schema.get("properties", {})
    final = {}

    for key in properties:
        # If the key is is_paid and it's missing or resolves to None, explicitly lock it to False
        if key == "is_paid" and (data.get(key) is None or data.get(key) == "null"):
            final[key] = False
        else:
            final[key] = data.get(key, None)

    return final
