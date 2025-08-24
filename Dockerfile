# syntax=docker/dockerfile:1

FROM python:3.13-slim-bookworm

ENV POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code

COPY . /code/

# ENV PYTHONPATH=/code

RUN poetry install --no-interaction --no-ansi

EXPOSE 8050

CMD [ "python", "neigh_ai/dashboard/app.py"]
