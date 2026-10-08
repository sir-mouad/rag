def check_args(query, k):
    if not isinstance(k, int) or k <= 0:
        print("Invalid Argument: --k must be a positive integer")
        return False
    if not query.strip():
        print("Invalid Argument: query must not be empty")
        return False
    return True