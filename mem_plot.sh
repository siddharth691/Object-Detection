#! /bin/bash
printf "Memory\n"

end=$((SECONDS+40500))
while [ $SECONDS -lt $end ]; do
MEMORY=$(top -bn1 | grep sagarwa+ | awk '{printf "%.2f%%\t\t\n", $(NF-2)}' | awk 'NR==1{printf "%.2f%%\t\t", $1}')
echo "$MEMORY"
sleep 405
done