from streamlit_authenticator.utilities.hasher import Hasher

passwords_to_hash = ['Alpha00@sliit']
hashed_passwords = Hasher(passwords_to_hash).generate()

print(hashed_passwords)