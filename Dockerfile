# Use the Amazon ECR Python 3.9 base image
FROM public.ecr.aws/python/python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the module files to the container
COPY . /app

# Install the module dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point for the container
ENTRYPOINT ["python", "/app/src/networks.py"]
