# this function converts a m by n matrix into a list of numbers in an anticlockwise direction
def unwrapMatrixEdge(mat:list[list[int]]) -> list[int]:
    # number of rows
    m = len(mat)
    # number of columns
    n = len(mat[0])
    
    # list of numbers in anticlockwise direction
    anticlockwise = []

    for i in range(m):
        anticlockwise.append(mat[i][0])
    
    for j in range(1, n):
        anticlockwise.append(mat[m - 1][j])

    for i in range(m - 2, -1, -1):
        anticlockwise.append(mat[i][n - 1])

    for j in range(n - 2, 0, -1):
        anticlockwise.append(mat[0][j])

    return anticlockwise


def unwrapMatrix(mat:list[list[int]]) -> list[int]:
    edge = unwrapMatrixEdge(mat)
    if len(mat) <= 2 or len(mat[0]) <= 2:
        return edge
    else:
        # remove the first and last row
        mat = mat[1:-1]
        # remove the first and last column
        for i in range(len(mat)):
            mat[i] = mat[i][1:-1]
        
        return edge + unwrapMatrix(mat)

def hopSkipJump(mat:list[list[int]]) -> int:
    unwrapped = unwrapMatrix(mat)
    # get last even index of unwrapped
    lastEvenIndex = len(unwrapped) - 1
    if lastEvenIndex % 2 == 1:
        lastEvenIndex -= 1
    return unwrapped[lastEvenIndex]

def main():
    matex1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print(unwrapMatrix(matex1))
    print(hopSkipJump(matex1))

    matex2 = [
        [1, 2],
        [3, 4],
    ]

    print(unwrapMatrix(matex2))
    print(hopSkipJump(matex2))

    matex3 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]

    print(unwrapMatrix(matex3))
    print(hopSkipJump(matex3))

    matex4 = [[1]]

    print(unwrapMatrix(matex4))
    print(hopSkipJump(matex4))

if __name__ == "__main__":
    main()
