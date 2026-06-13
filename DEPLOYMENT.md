# Self-RAG MCP AWS Deployment Journey

## From Local Prototype to Public AWS Deployment

### Project Goal

Deploy a customer-facing AI support application that combines:

* LangGraph multi-agent workflow
* SQL Agent (TiDB Cloud)
* RAG Agent (company documents)
* Human escalation workflow
* FastAPI backend
* Streamlit frontend

The objective was not just to build the application but to demonstrate the ability to take an AI prototype through deployment and make practical infrastructure decisions.

---

# Phase 1: Understanding the Starting Point

## Initial Architecture

Originally the project existed as:

```text
Local Machine

Streamlit UI
    |
    | Direct imports
    |
LangGraph
    |
    ├── TiDB Cloud
    ├── OpenRouter
    └── Gmail
```

The Streamlit application was importing and executing the LangGraph directly:

```python
from app.graph.builder import build_app
```

This meant:

* UI and backend were tightly coupled
* Streamlit executed the graph itself
* No clear API boundary existed
* Difficult to deploy professionally

At first this was not obvious because everything worked locally.

---

# Phase 2: Discovering the Production Problem

While discussing AWS deployment, an important realization occurred.

The Streamlit frontend was not acting as a frontend.

Instead it was acting as:

* UI
* Graph runner
* Business logic layer

all at the same time.

This created deployment problems:

```text
Browser
   |
Streamlit
   |
LangGraph
```

instead of:

```text
Browser
   |
Frontend
   |
API
   |
LangGraph
```

This became the first major architectural decision.

---

# Phase 3: Architecture Refactor

## Decision

Convert Streamlit into a thin client.

### Before

```python
build_app()
app.stream(...)
```

### After

```python
requests.post("/chat")
```

FastAPI became the single execution path.

New architecture:

```text
Streamlit
     |
 HTTP
     |
FastAPI
     |
LangGraph
```

Benefits:

* Single source of truth
* Easier deployment
* Easier scaling
* Cleaner architecture

---

# Phase 4: API Contract Expansion

While converting Streamlit to an API client, another issue was discovered.

The UI relied on internal graph state that was never exposed by the API.

Missing fields:

```python
rag_docs_used
sql_queries_executed
sql_result
handoff_summary
```

These existed in state but were not included in ChatResponse.

Solution:

Expanded the API contract so the UI could obtain everything via HTTP.

Lesson:

A frontend should never depend on backend internals.

---

# Phase 5: Local Validation

After refactoring:

### Tested

Health endpoint

```bash
curl http://localhost:8000/health
```

SQL flow

```bash
How many customers do you have?
```

Conversation flow

```bash
Hello
```

RAG flow

```bash
What does Bloomly do?
```

Escalation flow

```bash
Talk to human
```

All passed successfully.

---

# Phase 6: Streamlit Containerization

Originally only the API had a Docker image.

The Streamlit UI only worked locally.

Decision:

Create separate UI container.

Files added:

```text
Dockerfile.ui
requirements-ui.txt
```

Docker Compose updated:

```yaml
api:
  ...

ui:
  ...
```

Architecture became:

```text
Docker Compose

api
ui
```

Local validation confirmed:

* UI container worked
* API container worked
* Communication worked through Docker networking

---

# Phase 7: AWS Strategy Discussion

Several AWS deployment options were considered.

## Option 1

ECS Fargate

Pros:

* Fully managed
* Production ready

Cons:

* More complexity
* Higher monthly cost

## Option 2

EC2

Pros:

* Cheapest
* Simpler
* Good for portfolio demo

Cons:

* More operational responsibility

Decision:

Use EC2.

Reason:

Goal was demonstration and learning, not enterprise production scale.

---

# Phase 8: AWS Account Creation

A new AWS account was created.

Verification:

* AWS Free Plan active
* Promotional credits available
* No existing usage

Budget created:

```text
Monthly Budget = $5
```

Purpose:

Prevent accidental spending.

Lesson:

Always create billing safeguards first.

---

# Phase 9: EC2 Provisioning

Created:

```text
t3.small
```

Decision rationale:

t3.micro risked memory issues.

Project includes:

* PyTorch
* Sentence Transformers
* FAISS

Memory requirements were uncertain.

Chose safety over theoretical free tier limits.

Created:

* Key Pair
* Security Group

Security Group Rules:

```text
SSH 22 -> My IP

8501 -> Internet
```

---

# Phase 10: First SSH Connection

Verified:

```bash
ssh -i key.pem ec2-user@PUBLIC_IP
```

Successfully connected.

Validated:

```bash
docker --version
```

Docker already installed.

Validated:

```bash
docker ps
```

Docker permissions configured correctly.

---

# Phase 11: Repository Deployment

Installed Git.

Cloned repository:

```bash
git clone ...
```

Uploaded:

```text
.env
```

using SCP.

Verified:

```bash
ls -la .env
```

---

# Phase 12: First Major Failure

Attempted:

```bash
docker compose up -d
```

Received:

```text
compose build requires buildx 0.17.0 or later
```

This became the biggest infrastructure issue of the deployment.

---

# Phase 13: Buildx Investigation

Found:

```bash
docker buildx version
```

returned:

```text
0.12.1
```

Required:

```text
>=0.17.0
```

Decision:

Upgrade Buildx.

Installed:

```text
Buildx 0.17.1
```

Verification passed.

---

# Phase 14: Second Failure

Despite upgrading Buildx:

```bash
docker compose up -d
```

still failed.

Investigation showed:

```text
Buildx client = new
BuildKit backend = old
```

BuildKit version:

```text
0.12.x
```

Still insufficient.

---

# Phase 15: Builder Migration

Created new builder:

```bash
docker buildx create \
  --driver docker-container
```

Result:

```text
BuildKit 0.30.0
```

Verification:

```bash
docker buildx ls
```

showed:

```text
modernbuilder
BuildKit v0.30.0
```

Problem solved.

Lesson:

Buildx and BuildKit are separate components.

Upgrading one does not automatically upgrade the other.

---

# Phase 16: Successful Build

Executed:

```bash
docker compose up -d
```

Build completed:

```text
API image built
UI image built
Containers started
```

Images:

```text
API ≈ 2.5 GB
UI ≈ 780 MB
```

---

# Phase 17: Service Validation

Verified:

```bash
docker ps
```

Confirmed:

```text
API container running
UI container running
```

Verified logs:

```text
Application startup complete
Uvicorn running on port 8000
```

and

```text
Streamlit running on port 8501
```

---

# Phase 18: Public Access Validation

Accessed:

http://13.50.243.11:8501

Application loaded successfully.

Confirmed:

### Conversation

```text
Hello
```

Passed.

### SQL

```text
How many customers do you have?
```

Returned:

```text
99,441 customers
```

Passed.

### RAG

Document retrieval successful.

Passed.

### Escalation

Human handoff triggered.

Email delivered successfully.

Passed.

---

# Final Architecture

```text
Internet
    |
    v
AWS EC2 (t3.small)
    |
Docker Compose
    |
    +-- Streamlit UI
    |
    +-- FastAPI
            |
            +-- LangGraph
            +-- TiDB Cloud
            +-- OpenRouter
            +-- Gmail SMTP
```

---

# Key Lessons Learned

## Architecture

Do not allow the frontend to execute business logic.

Use:

```text
Frontend
  |
API
  |
Business Logic
```

---

## Docker

A local Docker build does not guarantee AWS deployment success.

Environment differences matter.

---

## AWS

Start simple.

EC2 was the correct choice for a first deployment.

Avoid premature migration to ECS.

---

## Troubleshooting

Always isolate the problem.

We solved:

1. Architecture issue
2. UI deployment issue
3. Buildx issue
4. BuildKit issue
5. Container deployment issue

one at a time.

---

# Final Outcome

Successfully deployed a customer-facing multi-agent AI application to AWS.

Capabilities demonstrated:

* LangGraph
* RAG
* SQL agents
* Human escalation
* FastAPI
* Docker
* Docker Compose
* AWS EC2
* Cloud networking
* Deployment troubleshooting
* Production validation

Result:

```text
Local Prototype
      ↓
API Refactor
      ↓
Containerization
      ↓
AWS Deployment
      ↓
Publicly Accessible AI Application
```

---

# Phase 19: Elastic IP Configuration

## Objective

Replace the temporary EC2 public IP with a permanent Elastic IP to ensure the application remains accessible after instance restarts.

## Actions Performed

1. Allocated a new Elastic IP in AWS EC2.
2. Associated the Elastic IP with the running EC2 instance.
3. Verified the association through the AWS console.
4. Confirmed the Elastic IP was attached to the correct instance.

## Elastic IP Assigned

```text
16.192.103.203
```

## Validation

Verified the EC2 instance networking section showed:

```text
Public IPv4 Address: 16.192.103.203
Private IPv4 Address: 172.31.15.105
```

## Outcome

Successfully established a static public IP address that would remain consistent across instance restarts.

---

# Phase 20: Custom Domain Deployment

## Objective

Expose the application through a professional domain name rather than a raw IP address.

## Challenge

The planned domain registration process was delayed due to domain approval and provisioning time.

## Decision

Use a free subdomain solution for immediate deployment and demonstration purposes.

## Selected Domain

```text
multiagent-ai.localnode.app
```

## DNS Configuration

Created an A record:

```text
Host: @
Type: A
Value: 16.192.103.203
```

## Validation

Executed:

```bash
nslookup multiagent-ai.localnode.app
```

Result:

```text
16.192.103.203
```

## Outcome

DNS successfully resolved to the EC2 instance.

---

# Phase 21: Initial Domain Access Failure

## Symptoms

Accessing:

```text
http://multiagent-ai.localnode.app
```

Resulted in:

```text
ERR_CONNECTION_TIMED_OUT
```

and later:

```text
ERR_SSL_PROTOCOL_ERROR
```

## Initial Assumptions

Potential causes considered:

1. DNS propagation delay
2. Security group misconfiguration
3. Streamlit accessibility issues
4. Elastic IP association issues
5. Nginx configuration issues

## Investigation

Verified:

### DNS

```bash
nslookup multiagent-ai.localnode.app
```

Returned:

```text
16.192.103.203
```

### Streamlit

```bash
curl http://localhost:8501
```

Returned Streamlit HTML successfully.

### Nginx Status

```bash
sudo systemctl status nginx
```

Confirmed Nginx running.

### Listening Ports

```bash
sudo ss -tulpn
```

Confirmed:

```text
80
8501
```

were listening.

## Outcome

Infrastructure appeared healthy but external access remained unavailable.

---

# Phase 22: Nginx Reverse Proxy Implementation

## Discovery

Inspection revealed:

```bash
sudo cat /etc/nginx/conf.d/*.conf
```

returned:

```text
No such file or directory
```

No reverse proxy configuration had been created.

## Solution

Created:

```text
/etc/nginx/conf.d/streamlit.conf
```

Configuration:

```nginx
server {
    listen 80;
    server_name multiagent-ai.localnode.app;

    location / {
        proxy_pass http://127.0.0.1:8501;

        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Validation

```bash
sudo nginx -t
```

returned:

```text
syntax is ok
```

## Outcome

Reverse proxy successfully configured.

---

# Phase 23: Nginx Virtual Host Conflict

## Symptoms

Despite the reverse proxy configuration, requests still returned:

```text
Welcome to nginx!
```

instead of the Streamlit application.

## Investigation

Executed:

```bash
curl -H "Host: multiagent-ai.localnode.app" http://localhost
```

and received the default Nginx landing page.

## Root Cause

The default Nginx server block was taking precedence over the Streamlit virtual host.

## Resolution

Modified configuration to become the default server:

```nginx
listen 80 default_server;
listen [::]:80 default_server;

server_name _;
```

Reloaded Nginx.

## Validation

```bash
curl http://localhost
```

returned Streamlit HTML.

## Outcome

Requests correctly routed to the application.

---

# Phase 24: External Connectivity Verification

## Investigation

Verified:

```bash
curl ifconfig.me
```

Result:

```text
16.192.103.203
```

Confirmed Elastic IP attachment.

Verified:

```bash
curl -I http://16.192.103.203
```

Result:

```text
HTTP/1.1 200 OK
```

Verified:

```bash
curl -I http://multiagent-ai.localnode.app
```

Result:

```text
HTTP/1.1 200 OK
```

## Outcome

HTTP deployment functioning successfully.

---

# Phase 25: SSL/TLS Enablement

## Objective

Provide secure HTTPS access.

## Installation

Installed Certbot:

```bash
sudo dnf install certbot python3-certbot-nginx -y
```

## Certificate Generation

Executed:

```bash
sudo certbot --nginx -d multiagent-ai.localnode.app
```

Successfully obtained:

```text
Certificate is saved at:
/etc/letsencrypt/live/multiagent-ai.localnode.app/fullchain.pem
```

## Automatic Renewal

Certbot automatically configured renewal scheduling.

## Validation

Executed:

```bash
curl -I https://multiagent-ai.localnode.app
```

Result:

```text
HTTP/1.1 200 OK
```

## Outcome

HTTPS deployment successfully completed.

---

# Final Production Deployment

## Public URL

```text
https://multiagent-ai.localnode.app
```

## Infrastructure Stack

* AWS EC2
* Docker
* Streamlit
* Nginx Reverse Proxy
* Elastic IP
* Custom DNS
* Let's Encrypt SSL
* Automatic Certificate Renewal

## Deployment Status

Production deployment completed successfully and accessible globally through HTTPS.
