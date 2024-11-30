FROM python:3.12

WORKDIR /app

RUN apt update

COPY /requirements.txt /app/requirements.txt

COPY .venv /app/.venv

RUN pip install -r requirements.txt

COPY . /app/

ENTRYPOINT [ "python" ]

CMD ["app.py"]