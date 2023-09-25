# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from the root directory into the container
COPY . /app/

# Install Jupyter Notebook and other dependencies
RUN pip install jupyter numpy pandas tqdm pillow ultralytics plotly firebase-admin opencv-python 

# Install Faiss (a vector similarity search library)
RUN pip install faiss-cpu

# Install libgl1-mesa-glx to resolve libGL.so.1 issue
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install libglib2.0-0 to resolve libgthread-2.0.so.0 issue
RUN apt-get update && apt-get install -y libglib2.0-0

RUN pip install openpyxl

# Install PyTorch and torchvision
RUN pip install torch torchvision

# Convert the Jupyter Notebook to a Python script
RUN jupyter nbconvert --to script artsee-demo.ipynb

EXPOSE 8080
# Run the converted Python script
CMD ["python", "artsee-demo.py"]
