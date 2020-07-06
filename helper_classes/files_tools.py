import string


def get_valid_file_name(original_file_name):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    new_valid_file_name = ''.join(
        c for c in original_file_name if c in valid_chars)
    return new_valid_file_name
