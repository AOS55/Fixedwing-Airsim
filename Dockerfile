FROM ubuntu:18.04

WORKDIR /app

COPY src/segmentation/deeplabv3_example.py ./
