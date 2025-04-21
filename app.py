from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List
import numpy as np
from datetime import datetime
from fastapi import FastAPI

app = FastAPI(
    title="Fantasy Football Prediction API",
    description="API for predicting fantasy football points and retraining the model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration
MODEL_PATH = "fantasy_model.joblib"
DATA_PATH = "fantasy_data.csv"  # Your cleaned fantasy football dataset
SCALER_PATH = "scaler.joblib"  # If you used feature scaling

# Load model and scaler at startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
else:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = None
    joblib.dump(model, MODEL_PATH)

# Define input/output schemas
class PlayerFeatures(BaseModel):
    player_id: str
    position: str  # QB, RB, WR, TE
    age: int
    experience: int
    avg_points_last_season: float
    team_strength: float  # 0-1 scale
    opponent_defense_rank: int  # 1-32
    home_game: bool
    weather_conditions: str  # 'good', 'bad'
    injury_status: str  # 'healthy', 'questionable', 'out'

class PredictionRequest(BaseModel):
    players: List[PlayerFeatures]

class PredictionResponse(BaseModel):
    player_id: str
    predicted_points: float
    confidence_interval: List[float]  # [lower_bound, upper_bound]

class RetrainRequest(BaseModel):
    retrain: bool = True
    test_size: float = 0.2  # Default test size for evaluation

class RetrainResponse(BaseModel):
    message: str
    training_date: str
    model_performance: dict
    feature_importances: dict

# Helper function to preprocess features
def preprocess_features(player: PlayerFeatures):
    # Convert categorical features
    features = {
        'position_QB': 1 if player.position == 'QB' else 0,
        'position_RB': 1 if player.position == 'RB' else 0,
        'position_WR': 1 if player.position == 'WR' else 0,
        'position_TE': 1 if player.position == 'TE' else 0,
        'age': player.age,
        'experience': player.experience,
        'avg_points_last_season': player.avg_points_last_season,
        'team_strength': player.team_strength,
        'opponent_defense_rank': player.opponent_defense_rank,
        'home_game': 1 if player.home_game else 0,
        'weather_bad': 1 if player.weather_conditions == 'bad' else 0,
        'injury_questionable': 1 if player.injury_status == 'questionable' else 0,
        'injury_out': 1 if player.injury_status == 'out' else 0
    }
    return pd.DataFrame([features])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fantasy Football Prediction API",
        "endpoints": {
            "/predict": "POST - Make predictions for players",
            "/retrain": "POST - Retrain the model with new data",
            "/features": "GET - List of expected features"
        }
    }

# Prediction endpoint
@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    try:
        predictions = []
        for player in request.players:
            # Preprocess features
            features_df = preprocess_features(player)
            
            # Scale features if scaler exists
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            pred = model.predict(features_scaled)[0]
            
            # Simple confidence interval estimation (can be improved)
            confidence = [max(0, pred - 3), pred + 3]  # ±3 points
            
            predictions.append({
                "player_id": player.player_id,
                "predicted_points": round(float(pred), 2),
                "confidence_interval": [round(x, 2) for x in confidence]
            })
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Retrain endpoint
@app.post("/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest):
    if not request.retrain:
        return {
            "message": "Retraining not requested",
            "training_date": datetime.now().isoformat(),
            "model_performance": {},
            "feature_importances": {}
        }
    
    try:
        # Load and prepare data
        data = pd.read_csv(DATA_PATH)
        X = data.drop(columns=['fantasy_points'])  # Adjust to your target column
        y = data['fantasy_points']
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )
        
        # Retrain model
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        
        # Evaluate model
        from sklearn.metrics import mean_absolute_error, r2_score
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(X.columns, model.feature_importances_))
        else:
            importances = {"message": "Feature importances not available for this model"}
        
        return {
            "message": "Model successfully retrained",
            "training_date": datetime.now().isoformat(),
            "model_performance": {
                "MAE": round(mae, 2),
                "R2_score": round(r2, 2),
                "test_size": request.test_size
            },
            "feature_importances": importances
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feature documentation endpoint
@app.get("/features")
async def feature_documentation():
    return {
        "features": {
            "player_id": "Unique player identifier",
            "position": "Player position (QB, RB, WR, TE)",
            "age": "Player age",
            "experience": "Years of NFL experience",
            "avg_points_last_season": "Average fantasy points per game last season",
            "team_strength": "Team strength rating (0-1 scale)",
            "opponent_defense_rank": "Opponent defense rank (1-32, 1=best)",
            "home_game": "Boolean indicating home game",
            "weather_conditions": "Game weather conditions ('good' or 'bad')",
            "injury_status": "Player injury status ('healthy', 'questionable', 'out')"
        }
    }