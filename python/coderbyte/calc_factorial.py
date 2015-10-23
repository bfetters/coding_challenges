def calc_factorial(num): 
    '''
    Calculate the factorial of a non-negative integer passed in as an argument.
    '''
    try: 
        num = int(num)
    except (TypeError, ValueError):
        print "Please enter a non-negative integer value."
        return None
    
    if num < 0:
        print "Please enter a non-negative integer value."
        
    result = 1
    for n in range(1,num + 1):
        result *= n
    
    return result
    
print "Please enter a non-negative integer: ",calc_factorial(raw_input())   