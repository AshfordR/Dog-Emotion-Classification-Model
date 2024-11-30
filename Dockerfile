FROM python:3.12

WORKDIR /app

RUN apt update
RUN pip install --no-cache-dir --upgrade pip

COPY /requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY . /app/

ENTRYPOINT ["python"]

CMD ["app.py"]