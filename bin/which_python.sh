#!/bin/bash
# which_python.sh
# Usage:
#   chmod +x ./bin/which_python.sh
#   ./bin/which_python.sh <username>
# Or (interactive):
#   ./bin/which_python.sh

user="$1"

# If no username passed as argument, prompt the user
if [ -z "$user" ]; then
    read -p "Enter the username: " user
fi

echo "Python processes run by user $user:"
echo "------------------------------------"
ps -u "$user" -o pid=,cmd= | grep -E 'python[0-9.]*[[:space:]]' | grep -v grep