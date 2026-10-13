#!/bin/bash
set -Eeuo pipefail

# Make ERR trap work inside functions + subshells
set -o errtrace

LAST_CMD=""
LAST_CONTEXT=""
LOG_FILE="${LOG_FILE:-}"   # set per script if you want log tail

notify_failure() {
  local lineno="$1"
  local exit_code="$2"
  local failed_cmd="${3:-}"

  local tail_txt=""
  if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
    tail_txt="$(tail -n 40 "${LOG_FILE}" || true)"
  fi

  if [[ -x ./bin/_notify_slack.sh ]]; then
    ./bin/_notify_slack.sh "FAILURE" \
"❌ Script failed
Script: $(basename "$0")
Host: $(hostname)
Line: ${lineno}
Exit code: ${exit_code}
Context: ${LAST_CONTEXT}
Failed cmd: ${failed_cmd}
Last command: ${LAST_CMD}
---- log tail (last 40 lines) ----
${tail_txt}"
  fi
}

on_error() {
  local lineno="$1"
  local exit_code="$2"
  local failed_cmd="$3"
  notify_failure "$lineno" "$exit_code" "$failed_cmd"
}

# Capture the command that failed via BASH_COMMAND
trap 'on_error "$LINENO" "$?" "$BASH_COMMAND"' ERR

# Wrapper to run commands while capturing context for Slack
run() {
  LAST_CONTEXT="$1"
  shift

  # Record the command as a printable string
  LAST_CMD="$(printf '%q ' "$@")"

  "$@"
}