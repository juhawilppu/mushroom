#!/bin/sh
# Publish output/site to Cloudflare Pages, as it stands.
#
# The Pages project's production branch is "main" while this repo's is
# master, so the branch is named here: left to itself, wrangler would read
# "master" from git and file every deploy as a preview. Wrangler is pinned
# because a bare `npx wrangler` resolves to whatever was published last,
# once a release still mid-publish that npm couldn't yet install.
set -eu
cd "$(dirname "$0")/.."

# The tiles aren't kept in git. Deployed without them -- from a fresh clone,
# say -- the live map would lose every stand.
if [ ! -d output/site/tiles ]; then
  echo "output/site/tiles is missing: run scripts/build_map.py first." >&2
  exit 1
fi

exec npx --yes wrangler@4.145.0 pages deploy output/site \
  --project-name mushroom --branch main "$@"
