import numpy as np

def _get_text_features(data):

    '''
        Args:
            str, input data
            
        Returns: 
            np.array, shape=(22,);
            an array of the text converted to features
            
    '''
    assert type(data) == str, f'Expected type {str}. Received {type(data)}.'

    data = r'{}'.format(data)

    n_lower = 0
    n_upper = 0
    n_digits = 0

    # make a mapping dict of special characters
    mapping_dict = {
        '-': 0, 
        '.': 1, 
        ',': 2, 
        '/': 3, 
        '\\': 4,
        ':': 5
    }
    initial_len_mapping_dict = len(mapping_dict)
    # add the alphabet as the keys to the mapping dict
    for idx, char in enumerate('abcedfghijklmnopqrstuvwxyz'):
        mapping_dict[char] = idx + initial_len_mapping_dict

    # get number of lower and upper case letters
    for char in data:
        if char.islower():
            n_lower += 1
        
        if char.isupper():
            n_upper += 1

        if char.isdigit():
            n_digits += 1

    # concat to form the vector in form:
    # | 0-29: character mapping | n_lower | n_upper | n_digits |

    vector_arr = np.zeros(35)
    for char in data.lower():
        if char in mapping_dict.keys():
            vector_arr[mapping_dict[char]] += 1
        else:
            pass

    vector_arr[32] = n_lower
    vector_arr[33] = n_upper
    vector_arr[34] = n_digits

    return vector_arr


if __name__ == "__main__":
    text = '\a'
    print(text)
    vec = _get_text_features(text)
    print('Specials: ', vec[:6])
    print('Characters: ', vec[6:32])
    print('Special', vec[32:])
