# 1. Base Image (Python 3.9 use panrom)
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install system dependencies (OpenCV-ku thevaiyanathu)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code (day8.py)
COPY day8.py .

# 6. Command to run the API
CMD ["uvicorn", "day8:app", "--host", "0.0.0.0", "--port", "8000"]