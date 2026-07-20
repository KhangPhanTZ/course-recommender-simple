# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Build the React/Vite app, then serve the static bundle with nginx.
# nginx also reverse-proxies /api -> the backend service so the SPA and API
# share an origin (no CORS) in the compose / ECS setup.
# ---------------------------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
# Build-time API base: the SPA calls same-origin /api, which nginx proxies.
ENV VITE_API_URL=/api
RUN npm run build


FROM nginx:1.27-alpine AS runtime
# Backend host:port that nginx proxies /api to (override in compose / ECS).
ENV API_UPSTREAM=api:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
# nginx:alpine auto-renders /etc/nginx/templates/*.template with envsubst on start.
