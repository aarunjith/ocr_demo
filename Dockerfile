FROM continuumio/miniconda3
WORKDIR /
ADD weights weights
ADD src src

RUN mkdir data
RUN mkdir templates
COPY requirements.txt .
COPY app.py .
COPY process_inputs.py .
COPY templates/w9_main.jpg templates/
COPY templates/w9_main.xml templates/

RUN pip3 install -r requirements.txt
RUN conda install -c conda-forge poppler
EXPOSE 8080

CMD ["uvicorn", "app:app", "--port", "8080"]