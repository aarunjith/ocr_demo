FROM continuumio/miniconda3
WORKDIR /
ADD weights weights
ADD src src

RUN mkdir data
RUN mkdir templates
COPY requirements.txt .
COPY app.py .
COPY tasks.py .
COPY process_inputs.py .
COPY templates/* templates/

RUN pip3 install -r requirements.txt
RUN conda install -c conda-forge poppler
EXPOSE 8080

CMD ["uvicorn", "app:app", "--port", "8080", "--host", "0.0.0.0"]