FROM tiangolo/uvicorn-gunicorn:python3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

COPY requirements.txt /root/.streamlit

WORKDIR /root/.streamlit

RUN pip install -r requirements.txt

COPY . /root/.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py"]