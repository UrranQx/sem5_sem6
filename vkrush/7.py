length = int(input())
array = input().split()
target = int(input())
array = [int(x) for x in array]


def exp_search(data, target, border):
    if border > len(data):
        return 'error border exceeded length of data'
    if data[border] == target:
        return border
    if data[border] < target:
        return exp_search(data, target, border * 2)
    else:
        return border // 2, border


print(*exp_search(array, target, 1))
# Чтобы делать бин поиск надо искать в диапазоне от exp_search[0] до exp_search[1]
