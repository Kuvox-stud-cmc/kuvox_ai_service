#!/bin/sh
set -eu

: "${REDIS_USERNAME:?REDIS_USERNAME is required}"
: "${REDIS_PASSWORD:?REDIS_PASSWORD is required}"
: "${REDIS_KEY_PATTERN:?REDIS_KEY_PATTERN is required}"
: "${REDIS_MAXMEMORY:?REDIS_MAXMEMORY is required}"

case "$REDIS_USERNAME$REDIS_PASSWORD$REDIS_KEY_PATTERN" in
  *' '*|*'
'*) echo "Redis ACL values must not contain whitespace" >&2; exit 1 ;;
esac

umask 077
mkdir -p /run/redis
printf '%s\n' \
  'user default off' \
  "user $REDIS_USERNAME on >$REDIS_PASSWORD ~$REDIS_KEY_PATTERN +@read +@write +ping -@dangerous +eval" \
  > /run/redis/users.acl

exec redis-server /etc/redis/redis.conf --maxmemory "$REDIS_MAXMEMORY"
