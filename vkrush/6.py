length = int(input())
array = input().split()
target = int(input())
array = [int(x) for x in array]


def bin_search(data, target, l, r):
    # print(f'l = {l} r = {r}')
    if l < 0 or r < 0:
        return -1
    if len(data) == 0:
        return -1

    mid = (l + r) // 2
    # print(f'mid = {mid}')
    midpoint = data[mid]
    # print(f'midpoint = {midpoint}')
    if midpoint == target:
        return mid
    if abs(r - l) == 0 and midpoint != target:
        if data[l] < target:
            return l + 1
        else:
            return l
    if midpoint > target:
        return bin_search(data, target, l, mid - 1)
    if midpoint < target:
        return bin_search(data, target, mid + 1, r)


print(bin_search(array, target, 0, len(array)))
