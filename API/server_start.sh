#!/bin/bash
echo "API_EXPLAINITALL_PYTHON"
uvicorn api_service_main:app --port 8000 --reload