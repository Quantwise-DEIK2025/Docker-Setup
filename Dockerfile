FROM huggingface/transformers-pytorch-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y curl && curl -fsSL https://ollama.com/install.sh | sh
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OLLAMA_HOST=http://ollama:11434