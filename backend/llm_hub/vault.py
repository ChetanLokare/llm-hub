import keyring
SERVICE = "LLMHub"

def save_secret(provider_id: str, secret: str):
    if not secret.strip():
        raise ValueError("API key cannot be empty.")
    keyring.set_password(SERVICE, provider_id, secret.strip())

def get_secret(provider_id: str):
    return keyring.get_password(SERVICE, provider_id)

def delete_secret(provider_id: str):
    try:
        keyring.delete_password(SERVICE, provider_id)
    except keyring.errors.PasswordDeleteError:
        pass
