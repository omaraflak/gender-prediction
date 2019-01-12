FROM python:3.6
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app
RUN pip install -r requirements.txt
EXPOSE 4000
COPY ./ /usr/src/app
CMD cd src/ && python server.py
