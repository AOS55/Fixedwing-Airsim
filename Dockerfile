FROM python:3.8.8

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

RUN ./python-test.sh

# ENV PORT=8080

# EXPOSE 8080

# CMD 