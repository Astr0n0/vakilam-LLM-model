from fastapi import Header, HTTPException

from config import API_KEY


def verify_api_key(
    x_api_key: str = Header(
        default=""
    )
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured."
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    return True
