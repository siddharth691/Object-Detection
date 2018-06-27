#! /bin/bash
printf "Memory\n"

end=$((SECONDS+40500))
while [ $SECONDS -lt $end ]; do
MEMORY=$(top -bn1 | grep 23479 | awk '{printf "%.2f%%\t\t\n", $(NF-2)}')
echo "$MEMORY"
sleep 405
done