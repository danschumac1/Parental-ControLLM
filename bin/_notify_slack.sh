#!/usr/bin/env bash
# chmod +x ./bin/_notify_slack.sh

# Usage:
#   _notify_slack.sh "STATUS" "MESSAGE"

STATUS="$1"
MESSAGE="$2"

python ./src/_ping_slack.py --message "[$STATUS] $MESSAGE"