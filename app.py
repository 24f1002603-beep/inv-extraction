import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )
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

    prompt = f"""
Extract the invoice details into a flat JSON object matching the exact schema provided below.

Rules:
- vendor: exact proper name as written (strip trailing punctuation).
- currency: valid 3-letter ISO 4217 code (e.g., USD, EUR, INR).
- total_amount: integer only. Convert words, text scales like K/M, and strip punctuation.
- invoice_date: format strictly as YYYY-MM-DD.
- due_in_days: integer count of days from the invoice date.
- is_paid: boolean.
- priority: exactly one of: low, normal, high, urgent.
- contact_email: lowercase email address.
- line_items: array of objects containing exactly {{sku, quantity, unit_price}}. Keep original order.
- item_count: integer count matching the length of line_items.

Schema:
{json.dumps(req.schema)}

Invoice Text:
{req.text}
"""

    try:
        # Using a highly accurate model for structured data tasks
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            temperature=0,
            max_tokens=1000,  # Bumped up so long line-items don't truncate the JSON
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction API. Return ONLY raw JSON matching the requested schema. No markdown formatting, no code blocks, no trailing conversational text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        text = response.choices[0].message.content
        data = json.loads(text)

        # Fix: Catch both missing keys AND empty strings
        if not data.get("contact_email"):
            match = re.search(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                req.text 
            )
            if match:
                data["contact_email"] = match.group(0).lower()

        # ---------------- Normalization & Type Safety ----------------

        if isinstance(data.get("vendor"), str):
            data["vendor"] = re.sub(
                r"[.,:;!?]+$",
                "",
                data["vendor"].strip()
            )

        if isinstance(data.get("contact_email"), str):
            data["contact_email"] = data["contact_email"].strip().lower()

        if isinstance(data.get("currency"), str):
            data["currency"] = data["currency"].strip().upper()

        if isinstance(data.get("priority"), str):
            data["priority"] = data["priority"].strip().lower()

        if data.get("total_amount") is not None:
            try:
                data["total_amount"] = int(float(data["total_amount"]))
            except Exception:
                pass

        if data.get("due_in_days") is not None:
            try:
                data["due_in_days"] = int(float(data["due_in_days"]))
            except Exception:
                pass

        if isinstance(data.get("line_items"), list):
            for item in data["line_items"]:
                if not isinstance(item, dict):
                    continue
                if item.get("quantity") is not None:
                    try:
                        item["quantity"] = int(float(item["quantity"]))
                    except Exception:
                        pass
                if item.get("unit_price") is not None:
                    try:
                        item["unit_price"] = int(float(item["unit_price"]))
                    except Exception:
                        pass
            
            data["item_count"] = len(data["line_items"])

        # -------- Ensure exact keys from schema are returned --------
        properties = req.schema.get("properties", {})
        final = {}
        for key in properties:
            final[key] = data.get(key)

        return final

    except Exception as e:
        print(f"Extraction Error: {e}")
        properties = req.schema.get("properties", {})
        return {key: None for key in properties}
