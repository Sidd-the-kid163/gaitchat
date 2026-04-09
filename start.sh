#!/bin/bash

# Start backend
cd backend
conda run -n base python main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "Waiting for backend..."
until curl -s http://localhost:8000/datasets > /dev/null; do
    sleep 1
done
echo "Backend ready"

# In production, frontend is already built and served by nginx
# This is only needed for local dev
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

# Handle shutdown - kill both on Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait