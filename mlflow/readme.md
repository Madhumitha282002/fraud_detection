# Project Title

One-line description of the project.

## Tech Stack

| Component   | Purpose                         | Port |
|------------|----------------------------------|------|
| MLflow     | Experiment tracking              | 5000 |
| Kafka      | Event streaming                  | 9092 |
| Zookeeper  | Kafka coordination               | 2181 |
| Redis      | Online store / feature cache     | 6379 |
| Kafdrop    | Kafka UI / debugging             | 9000 |

## Setup

### Prerequisites
- Docker
- Docker Compose

### Start services
```bash
docker compose up -d