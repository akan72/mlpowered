FROM python:3.9

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /frontend/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /frontend/.streamlit/credentials.toml'

WORKDIR /frontend

COPY . /frontend

RUN pip install -r requirements.txt

ENV IS_IN_CONTAINER TRUE

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py"]
