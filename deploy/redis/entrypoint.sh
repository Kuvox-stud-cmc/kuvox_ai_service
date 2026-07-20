#!/bin/sh
set -eu

: "${REDIS_API_PASSWORD:?REDIS_API_PASSWORD is required}"
: "${REDIS_AI_PASSWORD:?REDIS_AI_PASSWORD is required}"
: "${REDIS_MAXMEMORY:=256mb}"

case "$REDIS_API_PASSWORD$REDIS_AI_PASSWORD" in
  *' '*|*'
'*) echo "Redis ACL passwords must not contain whitespace" >&2; exit 1 ;;
esac

umask 077
mkdir -p /run/redis
printf '%s\n' \
  'user default off' \
  "user kuvox-api on >$REDIS_API_PASSWORD ~kuvox:v1:api:* +@read +@write +ping -@dangerous +eval" \
  "user kuvox-ai on >$REDIS_AI_PASSWORD ~kuvox:v1:ai:* +@read +@write +ping -@dangerous +eval" \
  > /run/redis/users.acl

exec redis-server /etc/redis/redis.conf --maxmemory "$REDIS_MAXMEMORY"
