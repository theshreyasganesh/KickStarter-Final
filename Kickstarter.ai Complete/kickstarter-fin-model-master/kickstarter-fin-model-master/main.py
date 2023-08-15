from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_pipeline
from model import avg_country_data
import uvicorn
# from app.model.model import __version__ as model_version

app = FastAPI()

class PredictionRequest(BaseModel):
    category: str
    total_funding: float
    country_code: str
    total_funding_rounds: int
    first_funding_date: str
    last_funding_date: str

class DataRequest(BaseModel):
    country_code: str

class PredictionOut(BaseModel):
    success: int

class DataOut(BaseModel):
    funding_total_average: float
    funding_duration_avg: float
    funding_rounds_avg: int

@app.get("/")
def home():
    return {"helath_check": "OK"}

@app.post("/predict",response_model=PredictionOut)
def predict(payload: PredictionRequest):
    success = predict_pipeline(payload.category, payload.total_funding, payload.country_code,
                               payload.total_funding_rounds, payload.first_funding_date, payload.last_funding_date)
    return {"success": success}

@app.post("/dataAvg",response_model=DataOut)
def predict(payload: DataRequest):
    arr = avg_country_data(payload.country_code)
    return {"funding_total_average": arr[0], "funding_duration_avg": arr[1], "funding_rounds_avg": arr[2]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)