FROM pytorch/pytorch

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

# RUN ./src/HPC_runs/python-test.sh

# ENV PORT=8080

# EXPOSE 8080

CMD python src/segmentation/deeplabv3_example.py