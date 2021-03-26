FROM pytorch/pytorch

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

RUN echo "export PYTHONPATH="/app/src"" >> ~/.bashrc

RUN /bin/bash -c "source ~/.bashrc"

CMD ["python", "src/segmentation/dataset_manager.py"]