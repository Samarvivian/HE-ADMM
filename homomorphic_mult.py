from __future__ import annotations

from typing import Any, Tuple


def _get_modulus_square(public_key: Any) -> int:
    """
    Extract n^2 from a public key. Supports objects with attributes or dicts.
    Prefers an explicit n_sq; otherwise derives from n if available.
    """
    n_sq = getattr(public_key, "n_sq", None)
    if isinstance(n_sq, int) and n_sq > 0:
        return n_sq

    n = getattr(public_key, "n", None)
    if isinstance(n, int) and n > 0:
        return n * n

    if isinstance(public_key, dict):
        if isinstance(public_key.get("n_sq"), int) and public_key["n_sq"] > 0:
            return int(public_key["n_sq"])  # type: ignore[index]
        if isinstance(public_key.get("n"), int) and public_key["n"] > 0:
            n_val = int(public_key["n"])  # type: ignore[index]
            return n_val * n_val

    raise ValueError("public_key must provide positive integer 'n_sq' or 'n'")


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Return (g, x, y) such that ax + by = g = gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m, assuming gcd(a, m) = 1."""
    a = a % m
    # Python 3.8+ supports pow(a, -1, m); use fallback for compatibility.
    try:
        return pow(a, -1, m)  # type: ignore[arg-type]
    except ValueError:
        g, x, _ = _extended_gcd(a, m)
        if g != 1:
            raise ValueError("modular inverse does not exist")
        return x % m


def homomorphic_mult(constant: int, cipher: int, public_key: Any) -> int:
    """
    Homomorphic multiplication by constant under Paillier: c ⊗ E(a) = E(c · a).

    In Paillier, multiplying plaintext by constant corresponds to raising the
    ciphertext to the power 'constant' modulo n^2:
        result = cipher^constant mod n^2

    Parameters
    ----------
    constant : int
        Scalar multiplier for the plaintext.
    cipher : int
        Ciphertext encrypting a.
    public_key : Any
        Public key providing either attribute/dict 'n_sq' or 'n' (to derive n^2).

    Returns
    -------
    int
        Ciphertext encrypting (constant · a).
    """
    if not isinstance(constant, int):
        raise TypeError("constant must be an integer")
    if not isinstance(cipher, int):
        raise TypeError("cipher must be an integer")

    n_sq = _get_modulus_square(public_key)
    c = cipher % n_sq

    if constant == 0:
        # E(0) under Paillier corresponds to 1 mod n^2 (neutral element for multiplication)
        return 1 % n_sq
    if constant > 0:
        return pow(c, constant, n_sq)

    # Negative constant: use modular inverse and positive exponent
    inv_c = _modinv(c, n_sq)
    return pow(inv_c, -constant, n_sq)


__all__ = ["homomorphic_mult"]




