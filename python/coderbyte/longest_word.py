def get_longest_word(sen): 
    '''
    Given a string, return the longest word withing importing
    any libraries (e.g. regex)
    '''
    # use only alpha characters to "string" sentence
    alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
    sen = ''.join([char for char in sen if char in alpha])
    
    long_word = ''
    for word in sen.split():
        if len(long_word) < len(word):
            long_word = word
            
    return long_word
    
print "Enter a sentence a I will return the longest word:\n",get_longest_word(raw_input())