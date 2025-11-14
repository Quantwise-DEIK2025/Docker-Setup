
# Quantwise Docker Setup

## Project Overview

Quantwise is a resource-efficient, localized AI research assistant designed to run on minimal hardware. The project enables you to deploy a fully functional AI research environment with just **8 GB of VRAM** and Docker.

## Requirements

- 16 GB RAM minimum
- 8 GB VRAM minimum
- Docker and Docker Compose installed

## Quick Start

0. Find the `know_how/` directory and upload any documents that you want your knowledge base to consist of. The intended use case is with PDF files, but we can handle many other text formats!
1. Navigate to the project root directory using a terminal
2. Run the following command:

```bash
docker compose up -d --build
```

3. Wait for the chunking process to complete
4. Open **OpenWebUI** (localhost:3000) and start chatting with your AI research assistant

## What happens after the first start?
When running `docker compose up --build`, it will take a long time, but don't worry, you only need to run it once!

You can stop the service after building by running
```bash
docker compose stop
```

And you can continue where you left off with
```bash
docker compose start
```

These commands should execute within a few seconds! Note: Our chunking script will look for any new files in the `know_how/` folder which were not present during the last run. If anything is found, it will chunk those documents too, so you can expand your knowledge base on the fly!

## Features

- **Lightweight**: Optimized for resource-constrained environments
- **Localized**: Runs entirely on your machine with no external dependencies
- **Easy Setup**: Single command deployment
- **OpenWebUI Integration**: User-friendly chat interface

## What Happens on First Run

When you execute `docker compose up`, the system will:
- Initialize the containerized environment
- Process and chunk your data
- Prepare the AI assistant for queries

Once chunking is complete, OpenWebUI will be ready for interaction.

## Notice

Please note that the first time you run `docker compose up`, it will take a really long time depending on your internet connection and hardware. The project needs to download approximately 30 GB of data on the first run, but is able to work entirely offline after that.