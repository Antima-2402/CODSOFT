import math

# Display the board
def print_board(board):
    for row in board:
        print("|".join(row))
        print("-" * 5)

# Check if there are any empty spots
def is_moves_left(board):
    for row in board:
        if " " in row:
            return True
    return False

# Evaluate the board and return a score
def evaluate(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2]:
            if row[0] == "X":
                return 10
            elif row[0] == "O":
                return -10

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == "X":
                return 10
            elif board[0][col] == "O":
                return -10

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == "X":
            return 10
        elif board[0][0] == "O":
            return -10

    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == "X":
            return 10
        elif board[0][2] == "O":
            return -10

    return 0

# Minimax algorithm
def minimax(board, depth, is_max):
    score = evaluate(board)

    if score == 10:
        return score - depth

    if score == -10:
        return score + depth

    if not is_moves_left(board):
        return 0

    if is_max:
        best = -math.inf

        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    best = max(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = " "
        return best
    else:
        best = math.inf

        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    best = min(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = " "
        return best

# Find the best move for the AI
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "X"
                move_val = minimax(board, 0, False)
                board[i][j] = " "

                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

if __name__ == "__main__":
    board = [[" " for _ in range(3)] for _ in range(3)]
    player_turn = True

    while is_moves_left(board) and evaluate(board) == 0:
        print_board(board)

        if player_turn:
            row, col = map(int, input("Enter your move (row and column): ").split())
            if board[row][col] == " ":
                board[row][col] = "O"
                player_turn = False
        else:
            move = find_best_move(board)
            board[move[0]][move[1]] = "X"
            player_turn = True

    print_board(board)
    score = evaluate(board)
    if score == 10:
        print("AI wins!")
    elif score == -10:
        print("You win!")
    else:
        print("It's a draw!")
