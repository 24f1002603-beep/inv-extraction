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

Return ONLY valid JSON.

Return EXACTLY the keys required by the supplied JSON Schema.

Do NOT return markdown.

Rules:

vendor
- Return exactly the company name.
- Remove trailing punctuation like '.', ',', ':'.
- Preserve the spelling.

currency
- Return ONLY ISO4217 code.
- Examples:
₹ -> INR
$ -> USD
€ -> EUR
£ -> GBP
¥ -> JPY

total_amount
- Integer only.
- Remove commas.
- Convert:
12K -> 12000
1,24,800 -> 124800
"Twelve thousand four hundred eighty" -> 12480

invoice_date
- YYYY-MM-DD only.

due_in_days
Examples:
Net 30 -> 30
Due in two weeks -> 14
Payable within 45 days -> 45

is_paid
true only if clearly paid.
Otherwise false.

priority
Only one of:
low
normal
high
urgent

contact_email
Lowercase only.

line_items
Maintain original order.

unit_price
Integer only.

item_count
Must equal len(line_items).

Schema:

{json.dumps(req.schema, indent=2)}

Invoice:

{req.text}

Return ONLY JSON.
"""

    try:

        response = client.chat.completions.create(
            model="openrouter/free",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        text = response.choices[0].message.content

        data = json.loads(text)

        # ---------- Normalization ----------

        if isinstance(data.get("vendor"), str):
            data["vendor"] = (
                data["vendor"]
                .strip()
                .rstrip(".,:;")
            )

        if isinstance(data.get("contact_email"), str):
            data["contact_email"] = (
                data["contact_email"]
                .strip()
                .lower()
            )

        if isinstance(data.get("currency"), str):
            data["currency"] = (
                data["currency"]
                .strip()
                .upper()
            )

        if isinstance(data.get("priority"), str):
            data["priority"] = (
                data["priority"]
                .strip()
                .lower()
            )

        if isinstance(data.get("line_items"), list):
            for item in data["line_items"]:
                if not isinstance(item, dict):
                    continue

                if "quantity" in item and item["quantity"] is not None:
                    try:
                        item["quantity"] = int(float(item["quantity"]))
                    except:
                        pass

                if "unit_price" in item and item["unit_price"] is not None:
                    try:
                        item["unit_price"] = int(float(item["unit_price"]))
                    except:
                        pass

            data["item_count"] = len(data["line_items"])

        if "total_amount" in data and data["total_amount"] is not None:
            try:
                data["total_amount"] = int(float(data["total_amount"]))
            except:
                pass

        if "due_in_days" in data and data["due_in_days"] is not None:
            try:
                data["due_in_days"] = int(float(data["due_in_days"]))
            except:
                pass

        # Return EXACTLY schema keys
        props = req.schema.get("properties", {})

        final = {}

        for key in props:
            final[key] = data.get(key)

        return final

    except Exception as e:

        print(e)

        props = req.schema.get("properties", {})

        return {
            key: None
            for key in props
        }
