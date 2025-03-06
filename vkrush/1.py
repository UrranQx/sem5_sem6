length = int(input())
string = input().split()
nums = [int(x) for x in string if x != '0']
for i in range(length - len(nums)):
    nums.append(0)

for element in nums:
    print(element, sep=' ', end=' ')
