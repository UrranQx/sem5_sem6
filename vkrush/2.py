length = int(input())
string = input().split()
element = input()
nums = [int(x) for x in string if x != element]
for element in nums:
    print(element, sep=' ', end=' ')
