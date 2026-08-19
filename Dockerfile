# Use python 3.11 as base image -------------->
FROM python:3.11-slim

# Set working directory 
WORKDIR /app

#Copy requirements and install dependencies -------->
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

#Copy rest of applicaion code 
COPY . . 

# Grant execution permission to the startup script
RUN chmod +x start.sh


#Expose the application port 
EXPOSE 8000 8501

# Run the startup script to launch both services
CMD ["./start.sh"]