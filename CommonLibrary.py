def try_parse_int(s, base=10, val=None):
  try:
    return True, int(s, base)
  except ValueError:
    return False, val