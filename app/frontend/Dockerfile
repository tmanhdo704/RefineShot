FROM node:20-alpine

WORKDIR /app

COPY app/frontend/package*.json ./
RUN npm ci

COPY app/frontend ./

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
