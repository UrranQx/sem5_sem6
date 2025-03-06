length = int(input())
array = input().split()
data = [int(x) for x in array]


# noinspection PyShadowingNames
def mysort(data):
    # noinspection PyShadowingNames
    def q_sort(array=data):
        if len(array) <= 1:
            return array
        if len(array) == 2:
            return [min(array), max(array)]
        pivot = array[(0 + len(array)) // 2]
        left = [x for x in array if x <= pivot]
        right = [x for x in array if x > pivot]
        return q_sort(left) + q_sort(right)

    return q_sort(data)


print(*mysort(data))
