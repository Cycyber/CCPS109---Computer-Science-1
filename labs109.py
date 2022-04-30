def ryerson_letter_grade(n):
    if n >= 90:
        return 'A+'
    elif n >= 85:
        return 'A'
    elif n >= 80:
        return 'A-'
    elif n >= 77:
        return 'B+'
    elif n >= 73:
        return 'B'
    elif n >= 70:
        return 'B-'
    elif n >= 67:
        return 'C+'
    elif n >= 63:
        return 'C'
    elif n >= 60:
        return 'C-'
    elif n >= 57:
        return 'D+'
    elif n >= 53:
        return 'D'
    elif n >= 50:
        return 'D-'
    else:
        return 'F'


def is_ascending(items):
    length = len(items)
    for i in range(length - 1):
        if items[i] >= items[i + 1]:
            return False
        elif items[i] < items[i + 1]:
            continue
    return True


def riffle(items, out=True):
    list1 = items[:len(items) // 2]
    list2 = items[len(items) // 2:]
    if out:
        return [a for b in zip(list1, list2) for a in b]
    else:
        return [a for b in zip(list2, list1) for a in b]


def only_odd_digits(n):
    nums = [int(x) for x in str(n)]
    real_nums = [a for a in nums if a % 2]
    if len(nums) == len(real_nums):
        return True
    else:
        return False


def is_cyclops(n):
    zero_count = str(n).count('0')
    if (len(str(n)) % 2 == 1) & (zero_count == 1) & (str(n)[len(str(n)) // 2] == '0'):
        return True
    else:
        return False


def domino_cycle(tiles):
    if len(tiles) == 0:
        return True
    elif len(tiles) == 1:
        return tiles[0][0] == tiles[0][1]
    elif len(tiles) > 1:
        for x in range(len(tiles)):
            if tiles[x][0] == tiles[x - 1][1]:
                continue
            else:
                return False
        return True


def colour_trio(colours):
    the_string = list(colours)
    while len(the_string) > 1:
        result = ''
        for i in range(0, len(the_string) - 1):
            if (the_string[i] == 'b' and the_string[i + 1] == 'r') or (
                    the_string[i] == 'r' and the_string[i + 1] == 'b') or (
                    the_string[i] == 'y' and the_string[i + 1] == 'y'):
                result = result + 'y'
            elif (the_string[i] == 'y' and the_string[i + 1] == 'b') or (
                    the_string[i] == 'b' and the_string[i + 1] == 'y') or (
                    the_string[i] == 'r' and the_string[i + 1] == 'r'):
                result = result + 'r'
            elif (the_string[i] == 'y' and the_string[i + 1] == 'r') or (
                    the_string[i] == 'r' and the_string[i + 1] == 'y') or (
                    the_string[i] == 'b' and the_string[i + 1] == 'b'):
                result = result + 'b'
        the_string = result
    return str(the_string[0])


def count_dominators(items):
    if not items:
        return 0
    items.reverse()
    bigger_item = items[0]
    dominator = 1
    for x in items:
        if x > bigger_item:
            bigger_item = x
            dominator += 1
    return dominator


def extract_increasing(digits):
    result = []
    current = 0
    previous = -1
    for i in range(len(digits)):
        a = int(digits[i])
        current = 10 * current + a
        if current > previous:
            result.append(current)
            previous = current
            current = 0
    return result


def words_with_letters(words, letters):
    result_list = []
    for word in words:
        wordc = 0
        letterc = 0
        while wordc != len(word) and letterc != len(letters):
            if word[wordc] == letters[letterc]:
                letterc += 1
            wordc += 1
        if letterc == len(letters):
            result_list.append(word)
    return result_list


def taxi_zum_zum(moves):
    x=0 
    direction = 0
    y=0
    if "F" in moves:
        for a in moves.split("F")[:-1]:
            direction = (a.count("R") - a.count("L") + direction) % 4
            x += 1 - abs(direction - 1)
            y += abs(direction - 2) - 1
    return x, y


def give_change(amount, coins):
    change = []
    coin_spot = 0
    while amount > 0:
        if amount >= coins[coin_spot]:
            amount -= coins[coin_spot]
            change.append(coins[coin_spot])
        else:
            coin_spot += 1
    return change


def safe_squares_rooks(n, rooks):
    columns = set(x for x in range(0, n))
    rows = set(x for x in range(0, n))
    for x, y in rooks:
        columns.discard(x)
        rows.discard(y)
        if len(rows) == 0 or len(columns) == 0:
            return 0
    return len(rows) * len(columns)


def pancake_scramble(text):
    for a in range(0, len(text) + 1):
        text = text[:a][::-1] + text[a:]
    return text


def words_with_given_shape(words, shape):
    # Compare every single word in words with the shape
    word_list = []
    for word in words:
        shape_numbers = []
        for i in range(0, len(word) - 1):
            if word[i + 1] > word[i]:
                shape_numbers.append(1)
            elif word[i + 1] < word[i]:
                shape_numbers.append(-1)
            else:
                shape_numbers.append(0)
        if shape_numbers == shape:
            word_list.append(word)
    return word_list


def is_left_handed(pips):
    pips_list = []
    left_hand_possibles = [[6, 4, 5], [1, 2, 3], [1, 5, 4], [3, 1, 2], [1, 4, 2], [2, 1, 4], [4, 2, 1], [1, 3, 5],
                           [5, 1, 3], [2, 3, 1], [4, 1, 5], [5, 4, 1], [6, 2, 4], [3, 2, 6], [6, 3, 2], [2, 4, 6],
                           [4, 6, 2], [2, 6, 3], [3, 6, 5], [5, 3, 6], [3, 5, 1], [4, 5, 6], [5, 6, 4], [6, 5, 3]]
    for i in pips:
        pips_list.append(i)
    if pips_list[:] in left_hand_possibles[:]:
        return True
    else:
        return False


def winning_card(cards, trump=None):
    ranks = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
             'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 'king': 13, 'ace': 14}
    trumpcount = 0
    position = 0
    if trump is None or trump not in [card[1] for card in cards]:
        trumpsuit = cards[0][1]
    else:
        trumpsuit = trump
        for i in range(len(cards)):
            if cards[i][1] == trumpsuit:
                trumpcount = 1
                position = i
        if trumpcount != 1:
            trumpsuit = cards[0][1]
    for i in range(len(cards)):
        if cards[i][1] == trumpsuit:
            if ranks[cards[i][0]] >= ranks[cards[position][0]]:
                position = i
    return cards[position]


def knight_jump(knight, start, end):
    knight = list(knight)
    position = []
    for i in range(len(start)):
        position.append(abs(start[i] - end[i]))
    position.sort()
    knight.sort()
    if position == knight:
        return True
    else:
        return False


def seven_zero(n):
    digits_in_n = 1
    result = 0
    while True:
        if n % 2 == 0 or n % 5 == 0:
            k = 1
            while k <= digits_in_n:
                x = int(k * '7' + (digits_in_n - k) * '0')
                if x % n == 0:
                    result = x
                    break
                k += 1
        else:
            x = int(digits_in_n * '7')
            result = x if x % n == 0 else 0
        digits_in_n += 1
        if result > 0:
            break
    return result


def can_balance(items):
    for i in range(len(items)):
        left_torque = 0
        right_torque = 0
        for j in range(0, i):
            left_torque += items[j] * (i - j)
        for k in range(i + 1, len(items)):
            right_torque += items[k] * (k - i)
        if left_torque == right_torque:
            return i
    return -1 #left torque didn't equal right torque


def reverse_ascending_sublists(items):
    # can also add to list and do list.reverse() or add to new list and list.clear()
    new_filtered = []
    filtered = []
    for item in items:
        if len(filtered) == 0:
            filtered.append(item)
        else:
            if item > filtered[0]:
                filtered.insert(0, item) #add bigger item at index 0 = Reverse list
            else:
                new_filtered.extend(filtered)
                filtered = [item]
    new_filtered.extend(filtered)
    return new_filtered


def frog_collision_time(frog1, frog2):
    # (sx,sy,dx,dy) so that (sx,sy) is the frog's starting position at the time zero, and
    # (dx,dy) is its constant direction vector for each successive hop.   (sx+t*dx,sy+t*dy)
    if frog1[2] - frog2[2] != 0:
        t = (frog2[0] - frog1[0]) // (frog1[2] - frog2[2])  # integer division, dont use fractions.fraction
    elif frog1[3] - frog2[3] != 0:
        t = (frog2[1] - frog1[1]) // (frog1[3] - frog2[3])
    else:
        return None
    temp_frog2y = frog2[1] + t * frog2[3]
    temp_frog1y = frog1[1] + t * frog1[3]
    temp_frog2x = frog2[0] + t * frog2[2]
    temp_frog1x = frog1[0] + t * frog1[2]
    if temp_frog1y == temp_frog2y and temp_frog1x == temp_frog2x and t >= 0:
        return round(t, 1)  # From class
    return None


def counting_series(n):
    # Champernowne word 1234567891011121314151617181920212223....
    x = 9
    sum_of_digits, position = 0, 0
    length_of_digits = 1
    while n - x * length_of_digits > 0:
        n -= x * length_of_digits
        sum_of_digits += x
        x *= 10
        length_of_digits += 1  # 1's vs 10's vs 100's 1000's 10,000's
    total = (n // length_of_digits) + sum_of_digits + 1
    position = n % length_of_digits
    return str(total)[position]


def josephus(n, k):
    result = []
    step_size = k
    positions = n
    position_list = [(i + 1) for i in range(n)]
    stop = 0
    while len(position_list) > 1:
        stop = (stop + (step_size - 1)) % positions
        positions -= 1
        result.append(position_list[stop])
        del position_list[stop]
    result.append(position_list[0])
    return result


def group_and_skip(n, out, ins):
    discarded_result = []
    while n != 0:
        discarded_result.append(n % out)  # stage 1
        n = n // out  # stage 2
        n = n * ins  # stage 3
    return discarded_result


def pyramid_blocks(n, m, h):
    # n*m gets bigger, while h gets lower
    # wolfram solution Sum[(n+i)(m+i), {i, 0, h-1}]
    spheres = 0
    while h > 0:
        spheres = spheres + (n * m)
        h -= 1
        n += 1
        m += 1
    return spheres


def count_growlers(animals):
    dogs, cats = [], []
    cat_count, dog_count = 0, 0
    growling_animals = 0
    index = 0
    length = len(animals)
    for animal in animals:
        if animal == 'dog' or animal == 'god':
            dog_count += 1
            dogs.append(dog_count)
            cats.append(cat_count)
        else:
            cat_count += 1
            dogs.append(dog_count)
            cats.append(cat_count)
    while index < len(animals):
        if animals[index] == 'dog' or animals[index] == 'cat':
            if index > 0 and dogs[index-1] > cats[index-1]:
                growling_animals += 1
        elif animals[index] == 'god' or animals[index] == 'tac':
            if (dogs[length-1] - dogs[index]) > (cats[length-1] - cats[index]):
                growling_animals += 1
        index += 1
    return growling_animals


def check_equality(piles, k):  # FROM TUTORIAL CLASS
    a_dict = {}
    for pile in piles:
        a_dict[pile] = 0
    if len(a_dict.keys()) == len(piles) and len(a_dict.keys()) == k:
        return True  # DONT WANT THIS OR ITS DONE
    return False


def bulgarian_solitaire(piles, k):
    if sum(piles) != k * (k + 1) / 2:  # Formula from ccps109.pdf
        return 0  # End with sum != k formula
    else:
        count = 0
        while not check_equality(piles, k):  # While NOT TRUE DO BELOW
            for i in range(len(piles)):
                piles[i] -= 1
            piles.append(len(piles))
            piles = [i for i in piles if i != 0]  # new piles list of non zero integers
            count += 1
    return count  # expected result


def scylla_or_charybdis(moves, n):
    min_step = 10000
    downfall_k = 1  # start at 0 steps away from center is downfall
    for a in range(1, len(moves) + 1):
        position = 0
        current = 0
        for b in range(a - 1, len(moves), a):
            if moves[b] == '+':  # go right
                position += 1
            elif moves[b] == '-':  # go left
                position -= 1
            current += 1
            if position == n or position == -n:
                if current < min_step:
                    min_step = current
                    downfall_k = a
    return downfall_k


def ztalloc(shape):
    previous_value = 1
    for i in range(len(shape) - 1, -1, -1):  # Start at end, work backwards
        if shape[i] == 'd':
            previous_value *= 2
        else:
            previous_value -= 1
            if previous_value % 3 != 0:  # not divisible by 3
                return None
            else:
                previous_value /= 3  # divide by 3
                if previous_value % 2 == 0:  # even number is bad
                    return None
    return int(previous_value)  # Return odd number


def collect_numbers(perm):
    inverse_perm = [0] * len(perm)  # create 0 list
    for i in range(len(perm)):
        inverse_perm[perm[i]] = i  # inverse list
    element = -5  # deep negative
    count = 1
    for x in range(len(inverse_perm)):
        if inverse_perm[x] > element:
            element = inverse_perm[x]
        else:
            count += 1
            element = inverse_perm[x]
    return count


from fractions import Fraction  # Required to make this work
def fractran(n, prog, giveup=1000):  # Done in Tutorial
    result = [n]
    while giveup > 0:
        for f in range(0, len(prog)):
            m = n * Fraction(prog[f][0], prog[f][1])
            if m - int(m) == 0:
                result.append(int(m))
                n = int(m)
                break
            elif f == len(prog):
                giveup = 0
                break
        giveup -= 1
    return result


def expand_intervals(intervals):
    if not intervals:  #EDGE CASE
        return []
    the_new_list = []
    splits = intervals.split(',')
    for x in splits:
        myrange = x.split('-')
        start_number = int(myrange[0])
        if len(myrange) == 1:
            the_new_list.append(start_number)
        else:
            while start_number <= int(myrange[1]):
                the_new_list.append(start_number)
                start_number += 1
    return the_new_list


def collapse_intervals(items):
    temp = []
    result = ''
    if not items:  # Edge case, empty list of items
        return ''
    if len(items) == 1:  # Edge case, 1 item in items
        return str(items[0])

    temp.append(items[0])
    previous = items[0]
    for x in items[1:]:
        if x == previous + 1:
            temp.append(x)
            previous = x
        else:
            if len(temp) == 1:
                result += str(temp[0])
                result += ','
                temp = [x]
                previous = x
            else:
                result += str(temp[0])
                result += '-'
                result += str(temp[len(temp) - 1])
                result += ','
                temp = [x]
                previous = x

    if len(temp) == 1:
        result += str(temp[0])
    else:
        result += str(temp[0])
        result += '-'
        result += str(temp[len(temp) - 1])

    return result


def three_summers(items, goal):  # can 3 integers from items equal to goal?
    for i in range(len(items)):  # 0 - 5  example   # iterate every possible until 1 is found
        x = items[i]             # 10
        start = i + 1            # 1
        end = len(items) - 1     # 4
        new_goal = goal - x       # 40-10=30
        while start < end:
            two_summers = items[start] + items[end]
            if two_summers == new_goal:
                return True
            elif two_summers < new_goal:
                start += 1
            else:
                end -= 1
    return False


def sum_of_two_squares(n):
    first_result = int(n ** 0.5)
    for a in range(first_result, 0, -1):
        temp = n - a * a
        if int(temp ** 0.5) ** 2 == temp and int(temp ** 0.5) > 0:
            return a, int(temp ** 0.5)


def count_carries(a, b):
    x = min(a, b)
    y = max(a, b)
    result = 0
    all_other_digits = 0
    while x != 0 or all_other_digits != 0:
        lowest_x = x % 10
        lowest_y = y % 10
        n = lowest_x + lowest_y + all_other_digits
        if n > 9:
            result += 1
            all_other_digits = n // 10
        else:
            all_other_digits = 0
        x = x // 10
        y = y // 10
    return result


def duplicate_digit_bonus(n):
    num = str(n)  # 43333
    score = 0
    k = 1
    x = num[0]
    for i in range(1, len(str(n))):  # 1-(5-1)
        if x == num[i]:
            k += 1
            if i + 1 == len(str(n)):
                score += 2 * 10 ** (k - 2)
        else:
            if k >= 2:
                score += 10 ** (k - 2)
            k = 1
        x = num[i]  # 4 3 3 3
    return score


def squares_intersect(s1, s2):
    # two squares as tuples (x1, y1, r1) and (x2, y2, r2)
    s1_corner_top_left = (s1[0], s1[1] + s1[2])
    s1_corner_bot_right = (s1[0] + s1[2], s1[1])
    s2_corner_top_left = (s2[0], s2[1] + s2[2])
    s2_corner_bot_right = (s2[0] + s2[2], s2[1])
    # if s1 top left is inside s2 bottom right or vice versa
    if (s1_corner_top_left[1] < s2_corner_bot_right[1]) or (s2_corner_top_left[1] < s1_corner_bot_right[1]) or \
            (s1_corner_top_left[0] > s2_corner_bot_right[0]) or (s2_corner_top_left[0] > s1_corner_bot_right[0]):
        return False
    return True


def remove_after_kth(items, k=1):
    dict = {}
    result = []
    if k == 0:
        return []  # empty list EDGE CASE
    for i in items:
        if i in dict:
            if dict[i] < k:
                dict[i] += 1  # Add to dict count to max of k values
                result += [i]
        else:
            dict[i] = 1
            result += [i]
    return result


def count_consecutive_summers(n):
   # https://en.wikipedia.org/wiki/Polite_number
    num_of_integers = 0
    for x in range(0, n):             # 0-42   X gets subtracted from n
        numerator = n - x * (x + 1) // 2    # 42-0*(0+1)//2=42
        if numerator <= 0:
          break  # end because we dont want any numbers at 0 or negative
        if numerator % (x + 1) == 0:
            num_of_integers += 1  # add to count of whole integers that work
    return num_of_integers


def first_preceded_by_smaller(items, k=1):
    # return first number in items that has k smaller before it
    for x in range(len(items)):
        count = 0
        for y in range(x):
            if items[x] > items[y]:
                count += 1  # increase count of items before kth item
        if count >= k:
            return items[x]  # return item at the spot


def eliminate_neighbours(items):
    if len(items) == 1:  # Edge Case
        return 1
    original_length = len(items)
    step_count = 0
    for i in range(1, len(items) + 1):
        if i in items:
            step_count += 1
            if len(items) == 1:
                items.pop(0)
                break
            center = items.index(i)
            left = center - 1
            if left < 0 or ((center + 1) < len(items) and items[center + 1] > items[left]):
                left = center + 1
            value = items[left]
            if center > left:
                center = left
            items.pop(center)
            items.pop(center)
            if value == original_length:
                break
    return step_count


def count_and_say(digits):
    result = ''
    if len(digits) == 0:
        return ''  # EDGE CASE
    else:
        character = digits[0]    # 3 from '333388822211177'
        count = 1                   # count start at 1
        for digit in digits[1:]:  # start digit 1 in from beginning
            if digit == character:  # increase count of 3 until next character isn't 3
                count += 1
            else:
                result += str(count)  # add count string to result string
                result += character    # add character to result string after count
                count = 1               #reset count to 1
            character = digit       # set character to new digit not the same as the last character
        result += str(count)
        result += character
        return result


def safe_squares_bishops(n, bishops):
    # the expression abs(r1-r2)==abs(c1-c2) checks that the horizontal distance
    # between those squares equals their vertical distance
    safe_cells = 0
    for row in range(n):
        for col in range(n):
            safe_spot = True
            for position in bishops:
                if abs(row - position[0]) == abs(col - position[1]):
                    safe_spot = False
                    break
            if safe_spot:
                safe_cells += 1
    return safe_cells


def reverse_vowels(text):
    vowels = ""
    vowel_position = 0
    result = ""
    for x in text:              # test   'Revorse the vewels'  'Reverse the vowels'
        if x in "aAeEiIoOuU":
            vowels = x + vowels  # put the vowels in the list in reverse order. push to begining of list
    if text[0] in "aAeEiIoOuU":
        if text[0].isupper():
            result += vowels[0].upper()    # Make vowels upper or lower in list as per text vowel position
        else:
            result += vowels[0].lower()
        vowel_position = 1
        text = text[1:]
    for one in text:
        if one in "aAeEiIoOuU":
            if one.isupper():
                result += vowels[vowel_position].upper()
            else:
                result += vowels[vowel_position].lower()
            vowel_position += 1
        else:
            result += one
    return result


from fractions import Fraction
from collections import deque


def calkin_wilf(n):
    sequence = deque()
    first = Fraction(1, 1)  # sequence, first, and append(first) all from problem outline
    sequence.append(first)
    i = 1
    while i < n:
        for _ in range(len(sequence)):
            front_fraction = sequence.popleft()
            front_numerator = front_fraction.numerator  # from problem outline
            front_denominator = front_fraction.denominator
            fraction_one = Fraction(front_numerator, (front_numerator + front_denominator))
            fraction_two = Fraction((front_numerator + front_denominator), front_denominator)

            sequence.append(fraction_one)  # from problem outline
            i += 1
            if i == n:
                return sequence[-1]

            sequence.append(fraction_two)  # from problem outline
            i += 1
            if i == n:
                return sequence[-1]


def postfix_evaluate(items):
    stack_list = []
    for x in items:
        if type(x) is int:          # go through each item in items list
            stack_list.append(x)    # if its an integer, append to end of list
            continue
        first = stack_list.pop()  # from problem description
        second = stack_list.pop()  # from problem description
        if x == '+':
            stack_list.append(second + first)
        elif x == '-':
            stack_list.append(second - first)
        elif x == '*':
            stack_list.append(second * first)
        elif x == '/':
            if first != 0:  # Edge case. Dont divide by 0  Case 15
                stack_list.append(second // first)
            else:
                stack_list.append(0)
    return stack_list.pop()


def subtract_square(queries):
    # https://en.wikipedia.org/wiki/Subtract_a_square
    result = []
    state = [False]  # base case, always start with cold/false
    for x in range(1, queries[len(queries) - 1] + 1):  # start after base case
        y = 1
        while True:
            if x - y * y < 0:
                state.append(False)
                break
            if not state[x - y * y]:
                state.append(True)
                break
            else:
                y += 1
    for x in range(0, len(queries)):  # add queries length of states to result
        result.append(state[queries[x]])
    return result


def brangelina(first, second):
    vowels = ['a', 'e', 'i', 'o', 'u']
    first_name_vgi = 0
    second_name_vgi = 0
    stop = False
    for i in range(len(first)):  # first name vowel group index
        if first[i] in vowels:
            if not stop:
                first_name_vgi = i
                stop = True
        else:
            stop = False
    for i in range(len(second)):  # second name vowel group index
        if second[i] in vowels:
            break
        second_name_vgi += 1
    stop = False
    for i in range(first_name_vgi - 1, -1, -1):
        if first[i] in vowels:
            first_name_vgi = i
            stop = True
        else:
            if stop:
                break
    return first[:first_name_vgi] + second[second_name_vgi:]  # return combined word


def line_with_most_points(points):
    slope_count = 0  # count points with similar slopes from dictionary of points and slope values
    if len(points) <= 2:  # Edge case, we need more than 2 points, or it's a line between them
        return len(points)
    for point_a in points:   # [(1, 4), (2, 6), (3, 2), (4, 10)]
        point_dict = {}
        duplicate = 0
        max_current = 0
        for point_b in points:
            if point_a != point_b:
                if point_a[0] == point_b[0]:
                    slope = 10000  # slope approaches infinity from both sides. Both points approach eachother
                else:
                    slope = float(point_b[1] - point_a[1]) / float(point_b[0] - point_a[0])  # calculate slope
                point_dict[slope] = point_dict.get(slope, 0) + 1
                max_current = max(max_current, point_dict[slope])
            else:
                duplicate += 1
        slope_count = max(slope_count, max_current + duplicate)
    return slope_count


def unscramble(words, word):
    new_list = [a for a in words if (a[0] == word[0] and a[-1] == word[-1] and len(word) == len(a))]
    result = []
    for x in new_list:
        # find words of equal length with same first and last character and same characters inside word
        if sorted(list(word[1:-1])) == sorted(list(x[1:-1])):
            result.append(x)
    return result


def count_divisibles_in_range(start, end, n):
    # NO LOOPS AT ALL!!!!
    # return len([x for x in range(start, end+1) if x%n == 0])   From problem pdf
    if start % n == 0:
        return (end - (start - n - start % n)) // n   # off by 1 edge case
    else:
        return (end - (start - n - start % n)) // n - 1


def frequency_sort(items):
    # frequency sort first, alphabetical sort 2nd if frequency is the same
    dict = {}  # dictionary of items and frequency
    for x in items:
        if x not in dict:
            dict[x] = 1
        else:
            dict[x] += 1
    new_list = sorted(items)  # sort alphabetically
    new_list = sorted(new_list, reverse=True, key=lambda x: dict[x])  # reverse sort so the highest frequency is first
    return new_list


def lunar_add(a, b):  # max function   REQUIRED FOR MULTIPLICATION OF 2> DIGITS
    first, second = str(a)[::-1], str(b)[::-1]
    sum = ''
    for i in range(max(len(first), len(second))):
        sum += max(first[i:i+1], second[i:i+1])
    return int(sum[::-1])
def lunar_multiply(a, b):  # min function
    first, second = str(a)[::-1], str(b)[::-1]
    big_digit = []
    answer=0  # start with 0
    for x in range(len(second)):
        d = ''
        for y in range(len(first)):
            d += min(first[y], second[x])
        big_digit.append('0'*x+d)   # add 0's to make both the same length
    for x in big_digit:  # only call this if length digits are 2+/carry
        answer = lunar_add(answer, x[::-1])
    return answer


def fibonacci_sum(n):
    first, second, sum, fib_sums, result = 0, 1, 0, [], []
    fib_sums.append(first)
    fib_sums.append(second)
    next = first + second
    while next <= n:
        fib_sums.append(next)
        first, second = second, next
        next = first + second
    fib_sums.sort(reverse=True)  # reverse list so big numbers are first
    while True:
        for i in fib_sums:
            if sum + i > n:  # fib_sums value + totalsum cannot be greater than n
                continue    # skip this fibsum in list, too high of value
            else:
                result.append(i)  # add the fibsum value to result list, its a winner
                sum += i          # add fibsum value to sum and recalculate max sum for next round
                if sum == n:     # sum is equal to n(max value). End the loops, its all done
                    break
                fib_sums.remove(i)
        if sum == n:  # sum is equal to n(max value). End the loops, its all done
            break
    return result


def oware_move(board, house):  # Entirely From Class
    stones, n = board[house], len(board)
    pstn = house
    board[house] = 0
    while stones > 0:
        j = 0
        for i in range(pstn + 1, len(board)):
            board[i] += 1
            j = i
            stones -= 1
            if stones == 0:
                break
        if stones > 0:
            for i in range(house):
                board[i] += 1
                stones -= 1
                if stones == 0:
                    break
        else:
            while board[j] in [2, 3] and j >= n / 2:
                board[j] = 0
                j -= 1
    return board


def arithmetic_progression(items):  # FROM CLASS
    total_length = len(items)
    items_set = set(items)
    if total_length == 1:
        return items[0], 0, 1
    current_max = []
    for (i, e) in enumerate(items):
        if len(current_max) >= total_length - i:
            break

        for y in range(i + 1, total_length):
            second_term = items[y]
            diff_in_val = second_term - e
            next_term = second_term + diff_in_val
            temp_list = [e, second_term]

            while next_term in items_set:
                temp_list.append(next_term)
                next_term += diff_in_val

            if len(temp_list) > len(current_max):
                current_max = temp_list
    return current_max[0], current_max[1] - current_max[0], len(current_max)


def candy_share(candies):
    # Game ends when everyone has 1 or 0 candies
    n = len(candies)  # list length
    turn_count = 0  # Example  [5, 1, 0, 0, 0, 0, 0, 1, 0] = 10 turns
    while True:  # RUN ONCE
        if not any(i >= 2 for i in candies):  # True if something still has 2 or higher in it
            return turn_count  # NO 2's, return turn count
        turn_count += 1
        new_candies = [0] * n
        for x in range(n):
            if candies[x] >= 2:  # if someone has 2 candies
                candies[x] -= 2  # take away 2 candies to give 1 to each person on either side
                if x == n - 1:  # Edge case, person is on far right side
                    new_candies[x - 1] += 1  # give chandy to person on left
                    new_candies[0] += 1  # give candy to person in position 0(start of list array)
                else:
                    new_candies[x - 1] += 1  # candy to person on your left
                    new_candies[x + 1] += 1  # candy to person on your right
            new_candies[x] += candies[x]  # add candy to the new position from old candy pile
        candies = new_candies  # update orignal list array and prepare to rerun while loop


import itertools
def count_troikas(items):
    # [42, 17, 42, 42, 42, 99, 42]
    # {42: [0, 2, 3, 4, 6], 17: [1], 99: [5]}
    dictionary = {}
    result = 0
    for item in items:
        if item not in dictionary:
            dictionary[item] = []
            for index in range(len(items)):
                if items[index] == item:
                    if index not in dictionary[item]:
                        dictionary[item].append(index)
    for key in dictionary:  # combinations/subsets  of dictionary[key] with length 2 below
        for i, j in itertools.combinations(dictionary[key], 2):
            k = j + (j - i)  # assumed third position
            if (k < len(items)) and items[k] == items[i]:  # required from problem documentation to increment count
                result += 1
    return result


def balanced_centrifuge(n, k):  # complete in tutorial, not my code
    if n < 2:
        return False
    factors, div = [], 2
    num = n
    while num > 1:
        if num / div % 1:
            div += 1
        else:
            if not div in factors:
                factors.append(div)
            num /= div
    def check(n, acc, factors, i):
        if acc == n:
            return True
        elif acc > n:
            return False
        else:
            for j in range(i, len(factors)):
                if check(n, acc + factors[j], factors, j):
                    return True
            return False
    return check(k, 0, factors, 0) and check(n - k, 0, factors, 0)

