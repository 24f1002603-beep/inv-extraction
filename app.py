import json
import os

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
You are an expert invoice extraction engine.

Extract information from the invoice text.

IMPORTANT:

Return ONLY valid JSON.

Return EXACTLY the JSON described by this schema.

Do NOT include extra keys.

Follow these rules:

- vendor exactly as written.
- currency must be ISO4217.
- total_amount must be an INTEGER.
- Convert 12K -> 12000.
- Convert 1,24,800 -> 124800.
- Convert amounts written in words into integers.
- invoice_date MUST be YYYY-MM-DD.
- due_in_days must be integer.
- "two weeks" -> 14.
- Net 30 -> 30.
- is_paid must be true/false.
- priority must be one of:
  low
  normal
  high
  urgent
- contact_email lowercase.
- unit_price integer.
- item_count equals length(line_items).

JSON Schema:

{json.dumps(req.schema, indent=2)}

Invoice Text:

{req.text}

Return ONLY JSON.
"""

    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            temperature=0,

            response_format={
                "type": "json_object"
            },

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.choices[0].message.content

        data = json.loads(text)

        return data

    except Exception as e:

        print(e)

        schema = req.schema

        props = schema.get("properties", {})

        return {
            key: None
            for key in props.keys()
        }