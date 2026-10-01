#!/bin/bash
# which_bash.sh
# Usage:
#   chmod +x ./bin/which_bash.sh
#   ./bin/which_bash.sh <username>
# Or (interactive):
#   ./bin/which_bash.sh

user="$1"

# If no username passed as argument, prompt the user
if [ -z "$user" ]; then
    read -p "Enter the username: " user
fi

echo "Bash processes run by user $user:"
echo "----------------------------------"
ps -u "$user" -o pid=,cmd= | grep -E 'bash( |$)' | grep -v grep