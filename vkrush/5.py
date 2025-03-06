length = int(input())
array = input().split()
even_nums = [int(x) for x in array if int(x) % 2 == 0]
if len(even_nums) == 0:
    print(-1)
else:
    print(even_nums[-1])
