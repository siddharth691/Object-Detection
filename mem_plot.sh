#! /bin/bash
printf "Memory\n"

end=$((SECONDS+60))
while [ $SECONDS -lt $end ]; do
MEMORY=$(top -bn1 | grep 26865 | awk '{printf "%.2f%%\t\t\n", $(NF-2)}')
echo "$MEMORY"
sleep 5
done