FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /backend

COPY requirements.txt /backend

WORKDIR /backend

RUN pip install -r requirements.txt

COPY . /backend

EXPOSE 8000

CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]