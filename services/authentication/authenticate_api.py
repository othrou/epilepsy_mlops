from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
import hashlib
from dotenv import load_dotenv
from fastapi.responses import Response




#################################### prometheus monitoring ########################################################
# Import Prometheus metrics
from prometheus_client import Counter, Histogram

# Load environment variables
load_dotenv()

# Prometheus Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])

#################################################################################################################

SECRET_KEY = "your_secret_key"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

fake_users_db = {
    "alice": {
        "username": "doc",
        "password": hash_password("secret") 
    }
}

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/login", response_model=Token)
def login(data: LoginRequest):
    user = fake_users_db.get(data.username)
    if not user or user["password"] != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/protected")
def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {current_user['username']}! This is protected."}

@app.post("/validate-token")
def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = fake_users_db.get(username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return {"valid": True, "username": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    


# Add a prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    # Retournez la réponse Prometheus avec le bon Content-Type
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)