from __future__ import annotations

from typing import Any


def _get_modulus_square(public_key: Any) -> int:
    """
    Extract n^2 from a public key. Supports objects with attributes or dicts.
    Prefers an explicit n_sq; otherwise derives from n if available.
    """
    # Attribute-style access (e.g., key.n_sq or key.n)
    n_sq = getattr(public_key, "n_sq", None)
    if isinstance(n_sq, int) and n_sq > 0:
        return n_sq

    n = getattr(public_key, "n", None)
    if isinstance(n, int) and n > 0:
        return n * n

    # Dict-style access (e.g., {"n_sq": ..., "n": ...})
    if isinstance(public_key, dict):
        if isinstance(public_key.get("n_sq"), int) and public_key["n_sq"] > 0:
            return int(public_key["n_sq"])  # type: ignore[index]
        if isinstance(public_key.get("n"), int) and public_key["n"] > 0:
            n_val = int(public_key["n"])  # type: ignore[index]
            return n_val * n_val

    raise ValueError("public_key must provide positive integer 'n_sq' or 'n'")


def homomorphic_add(cipher1: int, cipher2: int, public_key: Any) -> int:
    """
    Homomorphic addition under Paillier encryption: E(a) ⊕ E(b) = E(a + b).

    For Paillier, ciphertext-domain addition corresponds to multiplication mod n^2:
        result = (cipher1 * cipher2) mod n^2

    Parameters
    ----------
    cipher1 : int
        Ciphertext encrypting a.
    cipher2 : int
        Ciphertext encrypting b.
    public_key : Any
        Public key providing either attribute/dict 'n_sq' or 'n' (to derive n^2).

    Returns
    -------
    int
        Ciphertext encrypting (a + b).
    """
    if not isinstance(cipher1, int) or not isinstance(cipher2, int):
        raise TypeError("cipher1 and cipher2 must be integers")

    n_sq = _get_modulus_square(public_key)

    # Normalize into the residue class modulo n^2
    c1 = cipher1 % n_sq
    c2 = cipher2 % n_sq
    return (c1 * c2) % n_sq


__all__ = ["homomorphic_add"]




