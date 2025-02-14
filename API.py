import os
import joblib
import numpy as np
import pandas as pd
import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://phishing_feedback_db_user:YU0q5xSMwbvrMvgnMvZpjHnb4LRUGxAO@dpg-cundop23esus73cg5up0-a/phishing_feedback_db")

class URLInput(BaseModel):
    url: HttpUrl

class FeedbackInput(BaseModel):
    url: str
    is_phishing: bool
    feedback: bool  # True = confirm, False = opposite label

class PhishingDetectorAPI:
    def __init__(self, model_path='models/phishing_detector_v1.joblib'):
        # Load pre-trained model and preprocessing components
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoder = data['encoder']

        self.numeric_features = ['url_length', 'dots_count', 'digits_count', 'special_chars_count', 
                                 'path_depth', 'avg_token_length']
        self.categorical_features = ['has_http', 'has_www']

    def extract_url_features(self, url):
        url = str(url).lower()
        import re

        dots_count = url.count('.')
        path_depth = url.count('/')
        url_length = len(url)
        digits_count = sum(map(str.isdigit, url))
        special_chars_count = len(re.findall(r'[^a-z0-9.-_/]', url))
        
        tokens = re.split(r'[/.-_]', url)
        tokens = [token for token in tokens if token]
        avg_token_length = np.mean([len(token) for token in tokens]) if tokens else 0
        
        has_http = url.startswith('http://')
        has_www = 'www.' in url
        
        return {
            'url_length': url_length,
            'dots_count': dots_count,
            'digits_count': digits_count,
            'special_chars_count': special_chars_count,
            'path_depth': path_depth,
            'avg_token_length': avg_token_length,
            'has_http': int(has_http),
            'has_www': int(has_www)
        }

    def prepare_features(self, url_features):
        # Convert features to DataFrame
        df = pd.DataFrame([url_features])
        
        X_categorical = self.encoder.transform(df[self.categorical_features])
        X_numeric = self.scaler.transform(df[self.numeric_features])

        X_categorical_df = pd.DataFrame(X_categorical, columns=self.encoder.get_feature_names_out())
        X_numeric_df = pd.DataFrame(X_numeric, columns=self.numeric_features)

        return pd.concat([X_numeric_df, X_categorical_df], axis=1)

    def predict_phishing(self, url):
        url_features = self.extract_url_features(url)
        prepared_features = self.prepare_features(url_features)
        prediction = self.model.predict(prepared_features)[0]
        return bool(prediction)  # True = Phishing, False = Safe

# FastAPI Setup
app = FastAPI(title="Phishing URL Detector")
detector = PhishingDetectorAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/public", StaticFiles(directory=os.path.join(BASE_DIR, "public")), name="public")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "public"))

@app.on_event("startup")
async def startup():
    app.state.db = await asyncpg.create_pool(DATABASE_URL)
    await app.state.db.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE,
            label BOOLEAN
        )
    """)

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_url(input_url: URLInput):
    try:
        is_phishing = detector.predict_phishing(input_url.url)
        return {
            "url": str(input_url.url),
            "is_phishing": is_phishing,
            "risk_level": "High" if is_phishing else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ...existing code...

@app.post("/feedback")
async def store_feedback(feedback_data: FeedbackInput):
    try:
        await app.state.db.execute(
            "INSERT INTO feedback (url, label) VALUES ($1, $2) ON CONFLICT (url) DO UPDATE SET label = EXCLUDED.label",
            feedback_data.url, feedback_data.is_phishing
        )
        return {"message": "Feedback stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ...existing code...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
