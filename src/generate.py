import sys

def parseInput():
    show_prints = input("Show prints? (y/n): ").strip()
    if show_prints == "y":
        show_prints = True
    elif show_prints == "n":
        show_prints = False
    else:
        print("Invalid input")
        sys.exit(1)

    sequence = input("Sequence or range? (s/r): ").strip()
    direction = 1
    ending_value = 1
    if sequence == "s":
        starting_value = int(input("Enter the starting value: ").strip())
        direction = int(input("Direction? (-1, 1): ").strip())
    elif sequence == "r":
        starting_value = int(input("Enter the starting value: ").strip())
        ending_value = int(input("Enter the ending value: ").strip())
    else:
        print("Invalid input")
        sys.exit(1)
    
    return show_prints, sequence, direction, starting_value, ending_value

def recordData(value, steps, max_value, replace=False):
    if replace:
        with open("data/data.txt", "w") as f:
            f.write("In format of value, steps, maximum value reached\n")
    else:
        with open("data/data.txt", "a") as f:
            f.write(f"{value},{steps},{max_value}\n")

def collatzConjecture(n):
    steps = 0
    prev = n
    max_value = n
    while n != 1 and n != -1:
        if n % 2 == 0:
            n = n / 2
        else:
            n = 3 * n + 1
        if n > max_value:
            max_value = n
        if n == prev:
            return -1
        prev = n
        steps += 1
    return steps, int(max_value)

if __name__ == "__main__":
    show_prints, sequence, direction, starting_value, ending_value = parseInput()
    recordData(0, 0, 0, replace=True)

    if sequence == "r":
        for i in range(starting_value, ending_value + 1):
            steps, max_value = collatzConjecture(i)
            if show_prints:
                print(f"Value {i} reached 1 in {steps} steps")
            recordData(i, steps, max_value)
    else:
        value = starting_value
        while True:
            steps, max_value = collatzConjecture(value)
            if show_prints:
                if input(f"Value {value} reached 1 in {steps} steps (Press Enter to continue)") == "exit":
                    break
            recordData(value, steps, max_value)
            value += direction

