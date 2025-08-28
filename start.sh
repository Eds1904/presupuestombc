#!/usr/bin/env bash
set -e

mkdir -p .streamlit

# Render suele guardar PRIVATE_KEY con \n. Los convertimos a saltos reales:
GCP_PRIVATE_KEY_FIXED="$(printf '%b' "$GCP_PRIVATE_KEY")"

cat > .streamlit/secrets.toml <<EOF
[gcp]
type = "service_account"
project_id = "$GCP_PROJECT_ID"
private_key_id = "$GCP_PRIVATE_KEY_ID"
private_key = """$GCP_PRIVATE_KEY_FIXED"""
client_email = "$GCP_CLIENT_EMAIL"
client_id = "$GCP_CLIENT_ID"
token_uri = "https://oauth2.googleapis.com/token"

[drive]
folder_id = "$DRIVE_FOLDER_ID"
EOF

exec streamlit run wappmbc.py --server.port "$PORT" --server.address 0.0.0.0
