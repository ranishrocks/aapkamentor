# run_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import pickle

# --- 1. FastAPI app ---
app = FastAPI(title="Career Recommendation API", version="1.0")
MODEL_VERSION = "1.0"

# --- 2. Paths to your model ---
MODEL_PATH = "career_recommendation_model.pkl"
FEATURES_PATH = "model_features.pkl"

# --- 3. Load model safely ---
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        model_features = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    model = None
    model_features = []

# --- 4. Define Pydantic schemas ---
class Personality(BaseModel):
    analytical: float = Field(..., ge=0, le=1)
    creative: float = Field(..., ge=0, le=1)
    social: float = Field(..., ge=0, le=1)

class UserProfile(BaseModel):
    skills: List[str]
    interests: List[str]
    personality: Personality
    education: str
    experience: int

class CareerRecommendation(BaseModel):
    title: str
    confidence: int

class PredictionResponse(BaseModel):
    careers: List[CareerRecommendation]
    model_version: str

# --- 5. Preprocessing ---
education_mapping = {'High School':0,'Bachelor':1,'Master':2,'PhD':3}
technical_skills = ['Python','SQL','Java','Cloud Computing','Machine Learning','Statistics']
soft_skills = ['Communication','Leadership','Project Management']
creative_skills = ['Creative Writing','UI/UX']
business_skills = ['Business Strategy','Data Analysis','Excel']
career_names = ['Business Analyst','Data Scientist','Financial Analyst','Marketing Specialist',
                'Product Manager','Research Scientist','Software Engineer','UX Designer']

def preprocess(user: UserProfile, features: list) -> pd.DataFrame:
    df = pd.DataFrame(columns=features)
    df.loc[0] = 0
    for s in user.skills:
        if s in df.columns:
            df.loc[0,s] = 1
    for i in user.interests:
        if i in df.columns:
            df.loc[0,i] = 1
    # Cast explicitly to float to avoid FutureWarning
    df.loc[0,'analytical'] = float(user.personality.analytical)
    df.loc[0,'creative'] = float(user.personality.creative)
    df.loc[0,'social'] = float(user.personality.social)
    df.loc[0,'education_encoded'] = education_mapping.get(user.education,0)
    df.loc[0,'experience'] = user.experience
    df.loc[0,'tech_skill_count'] = sum(1 for s in user.skills if s in technical_skills)
    df.loc[0,'soft_skill_count'] = sum(1 for s in user.skills if s in soft_skills)
    df.loc[0,'creative_skill_count'] = sum(1 for s in user.skills if s in creative_skills)
    df.loc[0,'business_skill_count'] = sum(1 for s in user.skills if s in business_skills)
    return df

# --- 6. Confidence scoring ---
def predict_careers(user_df: pd.DataFrame):
    proba = model.predict_proba(user_df)
    scores = np.array([arr[:,1] for arr in proba]).T
    W_MODEL, W_HEURISTIC = 0.8, 0.2
    row = user_df.iloc[0]
    adj = {}
    for j, career in enumerate(career_names):
        base = scores[0,j]
        h = 0
        exp_boost = np.log1p(row.experience)*0.05
        if career=='Research Scientist' and row.education_encoded<2: h-=0.15
        if career=='Data Scientist' and row.tech_skill_count>=4: h+=0.1
        if career=='Software Engineer' and hasattr(row,'Java') and row.Java>0 and row.tech_skill_count>=3: h+=0.1
        if career=='Marketing Specialist' and row.creative>0.7 and row.social>0.7: h+=0.12
        blended = max(0,(base*W_MODEL)+(h*W_HEURISTIC))
        adj[career] = blended*(1+exp_boost)
    sorted_careers = sorted(adj.items(), key=lambda x:x[1], reverse=True)[:5]
    total = sum(score for _,score in sorted_careers)
    return [CareerRecommendation(title=c,confidence=int(round(s/total*100))) for c,s in sorted_careers]

# --- 7. API Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
def predict(user: UserProfile):
    if not model or not model_features:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    df = preprocess(user, model_features)
    recs = predict_careers(df)
    return PredictionResponse(careers=recs, model_version=MODEL_VERSION)

@app.get("/")
def root():
    return {"status":"ok","model_version":MODEL_VERSION}
