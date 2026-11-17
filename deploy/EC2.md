# Deploying to AWS EC2

The resume says "deployed on AWS EC2". Only keep that line if these steps are done.

## 1. Atlas cluster
- Create an Atlas cluster **on AWS**, in a region close to you (e.g. Mumbai `ap-south-1`). Note the region.
- Create a database user with readWrite on the `docsearch` database only.
- Run ingestion **from your laptop** against the Atlas URI (it's a one-time batch job; a micro instance is too small for it):
  ```bash
  MONGODB_URI='mongodb+srv://...' python -m app.indexes --wait
  MONGODB_URI='mongodb+srv://...' python -m app.ingest --dir data/pdfs --recursive
  MONGODB_URI='mongodb+srv://...' python scripts/stats.py
  ```

## 2. EC2 instance (same region as Atlas: the biggest latency win)
- Launch t3.micro (or t3.small), Amazon Linux 2023 or Ubuntu, **in the Atlas region**.
- Security group: 22 from *your IP only*; 80 and 443 from anywhere.
- Allocate an Elastic IP so the address survives reboots. Set a billing alarm.
- Atlas → Network Access: add the Elastic IP (not `0.0.0.0/0`).

## 3. Install Docker and run
```bash
# Amazon Linux 2023
sudo dnf install -y docker git && sudo systemctl enable --now docker && sudo usermod -aG docker ec2-user
sudo mkdir -p /usr/local/lib/docker/cli-plugins && sudo curl -sSL \
  https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose && sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# log out and back in, then:
git clone <your repo> atlas-doc-search && cd atlas-doc-search
cp .env.example .env && nano .env      # Atlas URI, OpenAI key, calibrated ABSTAIN_THRESHOLD
DOMAIN=your.domain docker compose -f deploy/docker-compose.prod.yml up -d --build
```
No domain? Leave `DOMAIN` unset: Caddy serves plain HTTP on port 80 (`http://<elastic-ip>`). A free
subdomain (e.g. DuckDNS) pointing at the Elastic IP gets automatic HTTPS.

`restart: unless-stopped` brings the containers back after crashes and reboots.

## 4. Verify and measure (from your laptop, not the server)
```bash
curl https://your.domain/health
python scripts/latency.py --url https://your.domain --n 200 --concurrency 1,5
python scripts/latency.py --url https://your.domain --endpoint ask --n 40 --concurrency 1
```
Record: EC2 region, Atlas region, instance type, date. The `/search` hybrid p95 (cold cache) is
the number to put on the resume. Measure `/ask` separately: it includes LLM generation and will be
well above 450 ms.
