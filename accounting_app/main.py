# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date, datetime
import io, os, re, json
from decimal import Decimal, InvalidOperation
import sqlite3
import uuid

# Ініціалізація FastAPI
app = FastAPI(
    title="JDG Accounting for PL (MVP)",
    description="AI-powered accounting system for Polish small businesses",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect('accounting.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            category TEXT,
            amount REAL NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tax_calculations (
            id TEXT PRIMARY KEY,
            year INTEGER NOT NULL,
            gross_income REAL NOT NULL,
            expenses REAL NOT NULL,
            tax_mode TEXT NOT NULL,
            zus_paid REAL DEFAULT 0,
            health_paid REAL DEFAULT 0,
            tax_result REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# ---------- Models ----------
class Transaction(BaseModel):
    id: Optional[str] = None
    date: date
    type: str = Field(..., regex="^(income|expense)$")
    category: str = "other"
    amount: float = Field(..., gt=0)
    description: str = ""

    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class TaxRequest(BaseModel):
    year: int = Field(..., ge=2020, le=2030)
    gross_income: float = Field(..., ge=0)
    expenses: float = Field(0.0, ge=0)
    tax_mode: str = Field(..., regex="^(progressive|flat19|lump_sum)$")
    zus_paid: float = Field(0.0, ge=0)
    health_paid: float = Field(0.0, ge=0)
    comment: Optional[str] = None

    @validator('expenses')
    def validate_expenses(cls, v, values):
        if 'gross_income' in values and v > values['gross_income']:
            raise ValueError('Expenses cannot exceed gross income')
        return v

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)

# ---------- Database Functions ----------
def get_db_connection():
    conn = sqlite3.connect('accounting.db')
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Utility Functions ----------
def explain(text: str):
    return {"explanation_simple": text}

# ---------- ZUS & Health calculation ----------
ZUS_DEFAULTS = {
    "monthly_social_min": 1000.0,
    "monthly_health_min": 300.0,
}

def calculate_zus_health_estimate(year: int, months_paid: int = 12):
    """
    Calculate ZUS contributions for the year
    """
    if not (1 <= months_paid <= 12):
        raise ValueError("Months paid must be between 1 and 12")
    
    ms = ZUS_DEFAULTS["monthly_social_min"]
    mh = ZUS_DEFAULTS["monthly_health_min"]
    
    total_social = round(ms * months_paid, 2)
    total_health = round(mh * months_paid, 2)
    
    return {
        "year": year,
        "months_paid": months_paid,
        "social_total": total_social,
        "health_total": total_health
    }

# ---------- Tax Calculation Logic ----------
def calculate_progressive_tax(taxable_income: Decimal) -> tuple[Decimal, str]:
    """Calculate progressive tax for Poland"""
    threshold = Decimal("120000.00")
    first_rate = Decimal("0.12")
    second_rate = Decimal("0.32")
    
    if taxable_income <= threshold:
        tax = taxable_income * first_rate
        comment = f"12% od dochodu do opodatkowania (do {threshold} PLN)"
    else:
        tax = threshold * first_rate + (taxable_income - threshold) * second_rate
        comment = f"12% do {threshold} PLN, 32% od nadwyżki"
    
    return tax.quantize(Decimal("0.01")), comment

def calculate_flat_tax(taxable_income: Decimal) -> tuple[Decimal, str]:
    """Calculate flat 19% tax"""
    rate = Decimal("0.19")
    tax = (taxable_income * rate).quantize(Decimal("0.01"))
    return tax, "Podatek liniowy 19% od dochodu"

def calculate_lump_sum_tax(taxable_income: Decimal) -> tuple[Decimal, str]:
    """Calculate lump sum tax (simplified)"""
    rate = Decimal("0.10")  # Simplified rate - actual rates vary by business type
    tax = (taxable_income * rate).quantize(Decimal("0.01"))
    return tax, "Ryczałt od przychodów ewidencjonowanych (stawka przykładowa 10%)"

# ---------- Endpoints: Transactions ----------
@app.post("/transactions", response_model=Transaction)
def add_transaction(transaction: Transaction):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    transaction_id = str(uuid.uuid4())
    
    cursor.execute('''
        INSERT INTO transactions (id, date, type, category, amount, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        transaction_id,
        transaction.date.isoformat(),
        transaction.type,
        transaction.category,
        transaction.amount,
        transaction.description
    ))
    
    conn.commit()
    conn.close()
    
    transaction.id = transaction_id
    return transaction

@app.get("/transactions", response_model=List[Transaction])
def list_transactions(limit: int = 100, offset: int = 0):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, date, type, category, amount, description 
        FROM transactions 
        ORDER BY date DESC 
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    transactions = []
    for row in cursor.fetchall():
        transactions.append(Transaction(
            id=row['id'],
            date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
            type=row['type'],
            category=row['category'],
            amount=row['amount'],
            description=row['description']
        ))
    
    conn.close()
    return transactions

@app.get("/transactions/{transaction_id}", response_model=Transaction)
def get_transaction(transaction_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM transactions WHERE id = ?', (transaction_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return Transaction(
        id=row['id'],
        date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
        type=row['type'],
        category=row['category'],
        amount=row['amount'],
        description=row['description']
    )

@app.delete("/transactions/{transaction_id}")
def delete_transaction(transaction_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM transactions WHERE id = ?', (transaction_id,))
    conn.commit()
    conn.close()
    
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {"message": "Transaction deleted successfully"}

# ---------- Endpoint: Tax calculations ----------
@app.post("/calculate_taxes")
def calculate_taxes(req: TaxRequest):
    try:
        # Convert to Decimal for precise calculations
        gross = Decimal(str(req.gross_income))
        expenses = Decimal(str(req.expenses))
        taxable_income = max(Decimal("0.0"), gross - expenses)
        zus_paid = Decimal(str(req.zus_paid))
        health_paid = Decimal(str(req.health_paid))

        # Calculate tax based on mode
        if req.tax_mode == "progressive":
            tax, comment = calculate_progressive_tax(taxable_income)
        elif req.tax_mode == "flat19":
            tax, comment = calculate_flat_tax(taxable_income)
        elif req.tax_mode == "lump_sum":
            tax, comment = calculate_lump_sum_tax(taxable_income)
        else:
            raise HTTPException(status_code=400, detail="Unknown tax_mode")

        tax_after_zus = max(Decimal("0.00"), tax - zus_paid)

        # Save calculation to database
        conn = get_db_connection()
        cursor = conn.cursor()
        calculation_id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO tax_calculations 
            (id, year, gross_income, expenses, tax_mode, zus_paid, health_paid, tax_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            calculation_id,
            req.year,
            float(req.gross_income),
            float(req.expenses),
            req.tax_mode,
            float(req.zus_paid),
            float(req.health_paid),
            float(tax_after_zus)
        ))
        conn.commit()
        conn.close()

        # Prepare response
        result = {
            "calculation_id": calculation_id,
            "year": req.year,
            "gross_income": float(gross),
            "expenses": float(expenses),
            "taxable_income": float(taxable_income),
            "tax_mode": req.tax_mode,
            "zus_paid": float(zus_paid),
            "health_paid": float(health_paid),
            "tax_before_zus": float(tax),
            "tax_after_zus": float(tax_after_zus),
            "plain_summary": f"Brutto: {float(gross):.2f} PLN. Koszty: {float(expenses):.2f} PLN. Dochód do opodatkowania: {float(taxable_income):.2f} PLN. Podatek: {float(tax):.2f} PLN.",
            **explain(comment)
        }

        # Add ZUS estimate
        zus_info = calculate_zus_health_estimate(req.year)
        result["zus_estimate"] = zus_info

        return result

    except InvalidOperation as e:
        raise HTTPException(status_code=400, detail="Invalid numeric values")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

# ---------- OCR endpoint ----------
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

@app.post("/ocr_receipt")
async def ocr_receipt(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not OCR_AVAILABLE:
        return {
            "ocr_available": False, 
            "message": "OCR not available. Install pytesseract and tesseract-ocr."
        }
    
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        
        # Convert image to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Perform OCR
        text = pytesseract.image_to_string(img, lang='pol+eng')
        
        # Extract amounts (PLN format)
        amounts = re.findall(r"\b\d+[.,]\d{2}\b", text)
        # Convert to float and replace comma with dot
        cleaned_amounts = [float(amount.replace(',', '.')) for amount in amounts]
        
        return {
            "ocr_available": True,
            "filename": file.filename,
            "raw_text": text.strip(),
            "amounts": cleaned_amounts,
            "largest_amount": max(cleaned_amounts) if cleaned_amounts else 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

# ---------- AI Chat endpoint ----------
@app.post("/ai_chat")
def ai_chat(payload: ChatRequest):
    message = payload.message.strip()
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_KEY:
        # Helpful responses without API key
        responses = {
            "podatek": "Mogę pomóc obliczyć podatki. Użyj endpointu /calculate_taxes z danymi: gross_income, expenses, tax_mode.",
            "zus": "Składki ZUS zależą od formy opodatkowania. Przybliżone obliczenia są dostępne w kalkulatorze podatkowym.",
            "ryczałt": "Ryczałt to uproszczona forma opodatkowania z stawkami 2-17% w zależności od rodzaju działalności.",
            "koszt": "Koszty uzyskania przychodu mogą obejmować materiały, wynajem, media, paliwo, etc.",
        }
        
        for keyword, response in responses.items():
            if keyword in message.lower():
                return {"assistant": response}
        
        return {
            "assistant": "Jestem asystentem księgowym. Mogę pomóc z: podatkami, ZUS, kosztami, ryczałtem. Zapytaj konkretnie!"
        }
    
    try:
        import openai
        openai.api_key = OPENAI_KEY
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """Jesteś pomocnym asystentem księgowym dla polskich małych firm (JDG). 
                    Odpowiadaj precyzyjnie i po polsku. Dotyczy: podatki VAT/CIT/PIT, ZUS, koszty, ryczałt, 
                    forma opodatkowania. Jeśli nie wiesz, polec konsultację z księgowym."""
                },
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return {"assistant": response.choices[0].message.content}
    
    except ImportError:
        return {"assistant": "OpenAI library not available. Install with: pip install openai"}
    except Exception as e:
        return {"assistant": f"Błąd połączenia z AI: {str(e)}"}

# ---------- Additional endpoints ----------
@app.get("/financial_summary")
def get_financial_summary(year: int = None):
    """Get financial summary for the year"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if year:
        cursor.execute('''
            SELECT 
                type,
                SUM(amount) as total,
                COUNT(*) as count
            FROM transactions 
            WHERE strftime('%Y', date) = ?
            GROUP BY type
        ''', (str(year),))
    else:
        cursor.execute('''
            SELECT 
                type,
                SUM(amount) as total,
                COUNT(*) as count
            FROM transactions 
            GROUP BY type
        ''')
    
    summary = {}
    for row in cursor.fetchall():
        summary[row['type']] = {
            'total': row['total'],
            'count': row['count']
        }
    
    conn.close()
    
    return {
        "year": year or "all",
        "summary": summary,
        "net_income": summary.get('income', {}).get('total', 0) - summary.get('expense', {}).get('total', 0)
    }

@app.get("/")
def root():
    return {
        "message": "JDG Accounting API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "transactions": "/transactions",
            "tax_calculations": "/calculate_taxes",
            "ocr": "/ocr_receipt",
            "ai_chat": "/ai_chat",
            "financial_summary": "/financial_summary"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected",
        "ocr_available": OCR_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)