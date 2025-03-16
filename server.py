# server.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from api.routes import router as api_router
from db.session import create_tables
from utils.logger import logger
from core.plugin_manager import plugin_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration
    logger.info("Configuration loaded")
    
    # Create database tables
    create_tables()
    
    # Initialize plugin manager
    await plugin_manager.initialize()
    
    logger.info("ManusPrime server started")
    
    yield
    
    # Cleanup plugins
    await plugin_manager.cleanup()
    
    logger.info("ManusPrime server shutting down")

# Create FastAPI app
app = FastAPI(
    title="ManusPrime",
    description="Multi-model AI agent API",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="web/templates")

# Create web routes
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Include API routes
app.include_router(api_router)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
