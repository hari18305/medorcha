# --- build stage ---
FROM node:20-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Build-time API base URL, threaded through from docker-compose.yml so
# it's never hardcoded here or in the source (see
# frontend/src/services/api.js). Defaults to the same value the app
# falls back to at runtime if unset.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# --- serve stage ---
FROM node:20-slim

WORKDIR /app

# `serve` is a small static file server -- avoids maintaining a separate
# nginx.conf for an app this size. No SPA fallback routing is needed:
# App.jsx does state-based screen switching, not URL-based routing.
RUN npm install -g serve@14

COPY --from=build /app/dist ./dist

EXPOSE 5173

CMD ["serve", "-s", "dist", "-l", "5173"]
