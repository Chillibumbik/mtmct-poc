#!/usr/bin/env bash
# One-time root setup on the server (run by the owner: `ssh -t onedash 'sudo bash ~/mtmct/deploy/server_root_setup.sh'`).
# 1) lets user `bogdan` run docker; 2) Let's Encrypt cert for the domain (HTTP-01 via the existing nginx on :80,
# certonly = existing sites are not modified); 3) nginx HTTPS site on :8443 with basic auth → 127.0.0.1:8090;
# 4) opens 8443 in ufw if ufw is active.
set -euo pipefail
# Deployment settings (DOMAIN etc.) live in the project's .env next to docker-compose.yml (not in git).
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
DOMAIN="${DOMAIN:?set DOMAIN in $ENV_FILE}"
PORT="${PORT:-8443}"
UPSTREAM="${UPSTREAM:-127.0.0.1:8090}"
APP_USER="${APP_USER:-mtmct}"

usermod -aG docker bogdan

if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  certbot certonly --nginx -d "$DOMAIN"   # interactive: e-mail / Let's Encrypt terms if no account yet
fi

if [ ! -f /etc/nginx/mtmct.htpasswd ]; then
  echo "Задайте пароль для входа в MTMCT (логин: $APP_USER):"
  printf '%s:%s\n' "$APP_USER" "$(openssl passwd -apr1)" > /etc/nginx/mtmct.htpasswd
  chmod 640 /etc/nginx/mtmct.htpasswd && chown root:www-data /etc/nginx/mtmct.htpasswd
fi

cat > /etc/nginx/sites-available/mtmct <<NGINX
server {
    listen $PORT ssl http2;          # 'http2 on' needs nginx >= 1.25.1
    listen [::]:$PORT ssl http2;
    server_name $DOMAIN;
    ssl_certificate     /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    auth_basic "MTMCT PoC";
    auth_basic_user_file /etc/nginx/mtmct.htpasswd;
    client_max_body_size 2g;          # custom multi-camera uploads
    location / {
        proxy_pass http://$UPSTREAM;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_buffering off;          # SSE progress streams
        proxy_read_timeout 1h;
        proxy_request_buffering off;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/mtmct /etc/nginx/sites-enabled/mtmct
if ! nginx -t; then rm -f /etc/nginx/sites-enabled/mtmct; echo 'nginx config invalid, site disabled' >&2; exit 1; fi
systemctl reload nginx

if ufw status 2>/dev/null | grep -q "Status: active"; then ufw allow "$PORT/tcp"; fi
echo "Готово: https://$DOMAIN:$PORT (логин $APP_USER). Перезайдите по ssh, чтобы группа docker применилась."
