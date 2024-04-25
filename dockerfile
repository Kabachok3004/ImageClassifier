FROM python:3.12-slim

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/test", "/results"]

ENTRYPOINT ["python", "main.py"]

CMD ["--input_dir", "/test", "--output_file", "/results/results.txt"]