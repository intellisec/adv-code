from hashlib import pbkdf2_hmac

def test():
  # ruleid: hashlib-pbkdf2-hmac-weak-iterations
  pbkdf2_hmac(x, y, z, 500)

def test2():
  # ruleid: hashlib-pbkdf2-hmac-weak-iterations
  pbkdf2_hmac(x, y, z, iterations=20)

def test3():
  # ok: hashlib-pbkdf2-hmac-weak-iterations
  pbkdf2_hmac(x, z, y, iterations=10000)

def test4():
  # ok: hashlib-pbkdf2-hmac-weak-iterations
  pbkf2_hmac(x,z,y, 1200)
