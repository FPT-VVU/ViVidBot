def find_first_list_from_response(response: str) -> str:
  # the response may not begin with a list, so we need to find the first list
  response = response.strip()
  response = response[response.index("[") :]
  return response
