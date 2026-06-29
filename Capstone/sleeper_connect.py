import requests

sleeper_error_codes = {
    400: "Error 400: Bad Request",
    404: "Error 404: Not Found",
    429: "Error 429: Too Many Requests",
    500: "Error 500: Internal Server Error",
    503: "Error 503: Service Unavailable",
}


def get_user_id(username):
    # this method returns the user_id based on an entered username
    # test using my sleeper username: brulism
    url = f"https://api.sleeper.app/v1/user/{username}"
    response = requests.get(url)
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: User not found"
    else:
        return response.json()["user_id"]


def get_leagues(user_id, season=2026):
    # since a user can be in multiple leagues,
    # this method returns a list of leagues they're in
    url = f"https://api.sleeper.app/v1/user/{user_id}/leagues/nfl/{season}"
    response = requests.get(url)
    league_info = {}
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: League not found"
    else:
        for league in response.json():
            league_info[league["name"]] = league["league_id"]
        return league_info


"""
what do we need for sleeper?
1. username (we can get their user_id and, in turn, their league id from this.
    they can also just copy paste the url from their league in 
    but for UX purposes maybe user id is easier)
    
    method: get_user_id(username) -> returns user_id

2. get_leagues, this is done and returns a list of leagues they're in

3. its only 2 methods for now, but to actually use it we 
    gotta do it in the actual app (chat says we should do a streamlit app)

"""
