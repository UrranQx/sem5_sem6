length = int(input())
array = input().split()
data = [int(x) for x in array]
counter = dict()
for element in data:
    if element in counter.keys():
        counter[element] += 1
    else:
        counter[element] = 1
# print(counter)
ans = -1
for entry in counter:
    if counter[entry] > length // 2:
        ans = entry

print(ans)
