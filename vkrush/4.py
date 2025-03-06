length = int(input())
array = input().split()
ans = 1e10
result = ''
for i in range(length - 1):
    t = abs(int(array[i]) - int(array[i + 1]))
    if t < ans:
        ans = t
        result = array[i] + ' ' + array[i + 1]
print(result)
